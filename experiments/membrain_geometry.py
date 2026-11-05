"""Membrain: continuation-region geometry and independent shadow calibration.

Run from the repository root with the environment used by corrected_benchmark.py.
Membership labels train only the shadow membership layer, never region classifiers.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import corrected_benchmark as base


def region_classifier(original, generated):
    """Separate original/generated decision boundary at each prefix.

    Auxiliary document splits stay intact. Class 1 is generated; logits are
    log(P(generated)/P(original)). Evaluation averages generation logits.
    """
    logits_o, logits_g, accuracies = [], [], []
    for k in range(4):
        train = torch.cat([original[200:300,k], generated[200:300,0,k]])
        y = torch.cat([torch.zeros(100),torch.ones(100)]).long()
        val = torch.cat([original[300:,k],generated[300:,0,k]])
        vy = torch.cat([torch.zeros(40),torch.ones(40)]).long()
        mean, sd = train.mean(0), train.std(0).clamp_min(.01)
        train, val = (train-mean)/sd, (val-mean)/sd
        model = nn.Sequential(nn.Linear(48,24),nn.ReLU(),nn.Linear(24,2))
        opt = torch.optim.Adam(model.parameters(),lr=.002,weight_decay=.001)
        best, saved, stale = float('inf'), None, 0
        for _ in range(80):
            for ids in torch.randperm(200).split(32):
                loss=F.cross_entropy(model(train[ids]),y[ids])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): value=F.cross_entropy(model(val),vy).item()
            if value < best-1e-5:
                best,saved,stale=value,{k:v.detach().clone() for k,v in model.state_dict().items()},0
            else: stale+=1
            if stale>=10: break
        model.load_state_dict(saved); model.eval()
        with torch.no_grad():
            accuracies.append((model(val).argmax(1)==vy).float().mean().item())
            o=model((original[:200,k]-mean)/sd)
            g=model((generated[:200,:,k]-mean)/sd)
            logits_o.append((o[:,1]-o[:,0]).numpy())
            logits_g.append((g[:,:,1]-g[:,:,0]).numpy())
    return np.stack(logits_o,1),np.stack(logits_g,2),accuracies


def model_run(seed, epochs, samples):
    torch.manual_seed(seed)
    docs,groups=base.corpus(seed)
    lm=base.LM()
    history=base.train_lm(lm,groups,epochs)
    torch.manual_seed(seed+10000)
    original,generated,loss=base.features(lm,docs,samples)
    o,g,acc=region_classifier(original,generated)
    # Whole-matrix Membrain boundary can capture cross-prefix dependencies that
    # the diagnostic per-prefix boundaries cannot. Preserve both comparisons.
    joint_gap,joint_acc=base.train_attack(original,generated,True)
    return dict(original_logits=o,generated_logits=g,loss=loss[:200],
                aux_accuracy=np.array(acc),lm_history=np.array(history),
                joint_gap=-joint_gap[:200],joint_aux_accuracy=np.array(joint_acc))


def matrix(record, method):
    o,g=record['original_logits'],record['generated_logits'].mean(1)
    if method=='Membrain trajectory CNN': return record['joint_gap'][:,None]
    if method=='Membrain mean gap': return (o-g).mean(1,keepdims=True)
    if method=='Membrain prefix gaps': return o-g
    if method=='Membrain original region': return o.mean(1,keepdims=True)
    if method=='LM loss': return record['loss'][:,None]
    raise ValueError(method)


def logistic(x,y):
    x=torch.as_tensor(x,dtype=torch.float64)
    y=torch.as_tensor(y,dtype=torch.float64)
    w=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([w,bias],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(x@w+bias,y)+.01*w.square().mean()
        loss.backward(); return loss
    opt.step(closure)
    return w.detach().numpy(),float(bias.detach())


def calibrate(logits,y):
    """Positive temperature + intercept: calibration cannot reverse score sign."""
    x=torch.as_tensor(logits,dtype=torch.float64); y=torch.as_tensor(y,dtype=torch.float64)
    scale=torch.zeros((),dtype=torch.float64,requires_grad=True)
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([scale,bias],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        a=F.softplus(scale)
        loss=F.binary_cross_entropy_with_logits(a*x+bias,y)+.001*(a-1).square()
        loss.backward(); return loss
    opt.step(closure)
    return float(F.softplus(scale).detach()),float(bias.detach())


def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-50,50)))


def ece(p,y):
    return sum(abs(p[mask].mean()-y[mask].mean())*mask.mean()
               for j in range(10) if (mask:=((p>=j/10)&(p<(j+1)/10 if j<9 else p<=1))).any())


def summarize(records,epochs,out):
    methods=['Membrain trajectory CNN','Membrain mean gap','Membrain prefix gaps','Membrain original region','LM loss']
    y=np.r_[np.ones(100),np.zeros(100)]
    train=[records[s] for s in range(31001,31005)]
    cal=[records[s] for s in range(32001,32004)]
    target=[records[s] for s in range(33001,33006)]
    metrics=[]; parameters={}; predictions={}
    for method in methods:
        x=np.concatenate([matrix(r,method) for r in train]); labels=np.tile(y,len(train))
        mean,sd=x.mean(0),np.maximum(x.std(0),1e-5)
        w,b=logistic((x-mean)/sd,labels)
        logits=lambda r: ((matrix(r,method)-mean)/sd)@w+b
        cx=np.concatenate([logits(r) for r in cal])
        a,c=calibrate(cx,np.tile(y,len(cal)))
        ps=np.stack([sigmoid(a*logits(r)+c) for r in target])
        predictions[method]=ps
        parameters[method]=dict(mean=mean.tolist(),sd=sd.tolist(),weights=w.tolist(),bias=b,
                                calibration_scale=a,calibration_bias=c)
        for group,ids in [('all',np.arange(100)),('ordinary',np.arange(80)),('strong',np.arange(80,100))]:
            values=[base.auc(p[ids],p[100:]) for p in ps]
            rng=np.random.default_rng(815)
            boot=[np.mean([base.auc(ps[i,rng.choice(ids,len(ids))],ps[i,rng.integers(100,200,100)])
                           for i in rng.integers(5,size=5)]) for _ in range(1000)]
            lo,hi=np.quantile(boot,[.025,.975])
            metrics.append(dict(epochs=epochs,method=method,group=group,auc=np.mean(values),
                ci_low=lo,ci_high=hi,brier=np.mean((ps-y)**2) if group=='all' else '',
                ece=ece(ps.ravel(),np.tile(y,5)) if group=='all' else ''))
    (out/f'calibrators_{epochs}.json').write_text(json.dumps(parameters,indent=2))
    np.savez_compressed(out/f'target_predictions_{epochs}.npz',**predictions)
    return metrics,predictions


def plots(records,epochs,predictions,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0072B2','#D55E00','#777777']
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']: fig.savefig(out/f'{name}_{epochs}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    targets=[records[s] for s in range(33001,33006)]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ids,title in [(axes[0],slice(0,80),'Ordinary members'),(axes[1],slice(80,100),'Strong members')]:
        # Average documents within model first; bands show across-model SD.
        for name,kind,sel,color in [('Member originals','o',ids,colors[0]),
                                  ('Nonmember originals','o',slice(100,200),colors[1]),
                                  ('Member-generated','g',ids,colors[0]),
                                  ('Nonmember-generated','g',slice(100,200),colors[1])]:
            values=np.stack([(sigmoid(r['original_logits'][sel]).mean(0) if kind=='o'
                              else sigmoid(r['generated_logits'][sel]).mean((0,1))) for r in targets])
            mu,sd=values.mean(0),values.std(0,ddof=1)
            ax.plot(base.PREFIXES,mu,'o-' if kind=='o' else 's--',color=color,label=name)
            ax.fill_between(base.PREFIXES,np.maximum(0,mu-sd),np.minimum(1,mu+sd),alpha=.08,color=color)
        ax.set_title(f'Membrain: {title.lower()}, {epochs} epochs')
        ax.set_xticks(base.PREFIXES); ax.set_xlabel('Prefix length'); ax.set_ylim(0,1)
    axes[0].set_ylabel('Region-classifier P(generated)\n(not membership probability)')
    axes[1].legend(fontsize=8); save(fig,'region_trajectories')
    fig,axes=plt.subplots(1,4,figsize=(15,3.5),sharex=True,sharey=True)
    for k,ax in enumerate(axes):
        for name,key,sel,color in [('Member original','original_logits',slice(0,100),colors[0]),
                                   ('Nonmember original','original_logits',slice(100,200),colors[1]),
                                   ('Generated (both)','generated_logits',slice(None),colors[2])]:
            # One mean generated probability per document avoids sample weighting.
            vals=np.concatenate([(sigmoid(r[key][sel])[:,k] if key=='original_logits'
                                  else sigmoid(r[key][sel]).mean(1)[:,k]) for r in targets])
            ax.hist(vals,bins=np.linspace(0,1,16),density=True,histtype='step',lw=1.8,label=name,color=color)
        ax.set_title(f'Prefix {base.PREFIXES[k]}'); ax.set_xlabel('P(generated region)')
    axes[0].set_ylabel('Density'); axes[-1].legend(fontsize=8)
    fig.suptitle(f'Membrain decision-region distributions: {epochs} epochs')
    save(fig,'region_distributions')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    y=np.tile(np.r_[np.ones(100),np.zeros(100)],5)
    for method,ps in predictions.items():
        p=ps.ravel(); xs=[]; ys=[]
        for j in range(10):
            mask=(p>=j/10)&(p<((j+1)/10) if j<9 else p<=1)
            if mask.sum()>=10: xs.append(p[mask].mean()); ys.append(y[mask].mean())
        axes[0].plot(xs,ys,'o-',label=method)
        axes[1].hist(p,bins=np.linspace(0,1,21),histtype='step',label=method,lw=1.5)
    axes[0].plot([0,1],[0,1],':',color='gray')
    axes[0].set(xlabel='Predicted membership probability',ylabel='Observed member fraction',xlim=(0,1),ylim=(0,1))
    axes[1].set(xlabel='Predicted membership probability',ylabel='Target document count')
    axes[1].legend(fontsize=8); fig.suptitle(f'Membrain held-out calibration: {epochs} epochs; membership prior 0.5')
    save(fig,'membership_calibration')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('results/membrain'))
    p.add_argument('--samples',type=int,default=16); args=p.parse_args()
    torch.set_num_threads(2)
    os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME','/workspace/.cache')
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    allmetrics=[]
    for epochs in [5,60]:
        records={}
        for seed in [*range(31001,31005),*range(32001,32004),*range(33001,33006)]:
            path=out/f'model_{epochs}_{seed}.npz'
            if path.exists():
                with np.load(path) as f: records[seed]={k:f[k] for k in f.files}
                if (records[seed]['generated_logits'].shape[1]!=args.samples
                        or len(records[seed]['lm_history'])!=epochs):
                    raise ValueError('Cached settings differ; use a new --output directory.')
                if 'joint_gap' not in records[seed]:
                    print(f'Membrain joint-matrix extension epochs={epochs} seed={seed}',flush=True)
                    records[seed]=model_run(seed,epochs,args.samples)
                    np.savez_compressed(path,**records[seed])
            else:
                print(f'Membrain model epochs={epochs} seed={seed}',flush=True)
                records[seed]=model_run(seed,epochs,args.samples)
                np.savez_compressed(path,**records[seed])
        metrics,preds=summarize(records,epochs,out); allmetrics+=metrics
        plots(records,epochs,preds,out)
        print(f'Membrain finished {epochs} epochs',flush=True)
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(allmetrics[0])); w.writeheader(); w.writerows(allmetrics)
    (out/'config.json').write_text(json.dumps(dict(samples=args.samples,budgets=[5,60],
        train_seeds=list(range(31001,31005)),calibration_seeds=list(range(32001,32004)),
        target_seeds=list(range(33001,33006)),membership_prior=.5,
        membership_mix={'ordinary':.8,'strong':.2},torch=torch.__version__,numpy=np.__version__),indent=2))
    print('DONE',flush=True)


if __name__=='__main__': main()
