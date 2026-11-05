"""Reviewer-directed Membrain study: common support, loss control, and ablations.

Origin networks never see membership labels. Separate, explicitly supervised
shadow probes test whether their scores add information beyond flexible LOSS.
Fresh target sentences/models are excluded from previous experiments.
"""
import argparse
import copy
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import membrain_recovery as rec
import membrain_trajectory_classifier as origin
from membrain_loss_curves import auc
from membrain_geometry import calibrate

ROOT=Path(__file__).resolve().parents[1]
VERSION='reviewer_beyond_loss_v1'
BUDGETS=[1,5,20]
SEEDS={'train':list(range(61001,61005)),'validation':list(range(62001,62003)),
       'target':list(range(63001,63005))}
VARIANTS=['cnn','flat','shuffle','pooled','loss_origin']
PROBES=['LOSS','LOSS + CNN','LOSS + flat','LOSS + shuffle','LOSS + pooled',
        'LOSS + loss-origin','LOSS + loss curves','LOSS + recovery',
        'LOSS + statistics','LOSS + statistics + CNN']
CALIPERS=[.01,.02,.05]
LAMBDAS=[.001,.01,.1]
EXTRA_PROBE='LOSS + learned representation'
ANCHOR_PROBE='LOSS + anchor representation'


def fresh_pools():
    old=json.loads((origin.RECOVERY/'sentences.json').read_text())
    old_aux=json.loads((ROOT/'results/membrain_trajectory_classifier/auxiliary_sentences.json').read_text())
    excluded=set(old['evaluation']+old['background']+sum(old_aux.values(),[]))
    data=rec.previous.SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest()==rec.previous.SOURCE_SHA
    text=re.sub(r'\s+',' ',data.decode());sentences=[];seen=set(excluded)
    for part in re.split(r'(?<=[.!?])\s+',text):
        part=re.sub(r'^[A-Za-z][A-Za-z ]{0,35}:\s*','',part).strip()
        part=''.join(c if c in rec.ALPHABET else ' ' for c in part)
        part=re.sub(r'\s+',' ',part)
        if 80<=len(part)<=120 and part[-1:] in '.!?' and part not in seen:
            seen.add(part);sentences.append(part)
    assert len(sentences)>=1216,len(sentences)
    chosen=np.random.default_rng(81109).permutation(len(sentences))[:1216]
    selected=[sentences[i] for i in chosen]
    pools=dict(background=selected[:128],auxiliary=selected[128:448],
               train=selected[448:704],validation=selected[704:960],target=selected[960:1216])
    assert len(set(sum(pools.values(),[])))==1216
    assert not set(selected)&excluded
    return pools,len(excluded)


class Network(nn.Module):
    def __init__(self,variant,d=96,k=12):
        super().__init__();self.variant=variant
        if variant in ['cnn','shuffle']:
            self.body=nn.Sequential(nn.Conv1d(d,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,16,3,padding=1),nn.ReLU())
            self.head=nn.Sequential(nn.Flatten(),nn.Linear(k*16,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,1))
        elif variant in ['flat','loss_origin']:
            width=15 if variant=='flat' else 32
            self.head=nn.Sequential(nn.Flatten(),nn.Linear(k*d,width),nn.ReLU(),nn.Dropout(.1),nn.Linear(width,1))
        else:self.head=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,1))

    def forward(self,x):
        if self.variant=='shuffle' and self.training:
            idx=torch.rand(len(x),x.shape[1]).argsort(1)
            x=x.gather(1,idx[:,:,None].expand_as(x))
        if self.variant in ['cnn','shuffle']:return self.head(self.body(x.transpose(1,2)))[:,0]
        return self.head(x.mean(1) if self.variant=='pooled' else x)[:,0]


def logits(net,x,bank):
    if net.variant!='shuffle':return net(x)
    return torch.stack([net(x[:,p]) for p in bank]).mean(0)


def fit_origin(original,generated,variant,seed):
    torch.manual_seed(seed);d=original.shape[-1]
    a,b=original[:160],generated[:160,1:]
    mean=.5*(a.mean((0,1))+b.mean((0,1,2)))
    second=.5*((a*a).mean((0,1))+(b*b).mean((0,1,2)))
    sd=np.sqrt(np.maximum(second-mean*mean,1e-4))
    o=torch.tensor((original-mean)/sd,dtype=torch.float32)
    g=torch.tensor((generated[:,1:]-mean)/sd,dtype=torch.float32)
    bank=torch.stack([torch.randperm(12) for _ in range(8)])
    net=Network(variant,d);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.01)
    best,state,stale=float('inf'),None,0
    for _ in range(120):
        net.train()
        for ids in torch.randperm(160).split(32):
            lo=net(o[ids]);lg=net(g[ids].reshape(-1,12,d))
            loss=.5*(F.binary_cross_entropy_with_logits(lo,torch.zeros_like(lo))+
                      F.binary_cross_entropy_with_logits(lg,torch.ones_like(lg)))
            opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            lo=logits(net,o[160:200],bank);lg=logits(net,g[160:200].reshape(-1,12,d),bank)
            value=.5*(F.binary_cross_entropy_with_logits(lo,torch.zeros_like(lo))+
                       F.binary_cross_entropy_with_logits(lg,torch.ones_like(lg)))
        if value.item()<best-1e-5:best,state,stale=value.item(),copy.deepcopy(net.state_dict()),0
        else:stale+=1
        if stale>=15:break
    net.load_state_dict(state);net.eval()
    with torch.no_grad():co=logits(net,o[200:240],bank).double();cg=logits(net,g[200:240].reshape(-1,12,d),bank).double()
    t=torch.zeros((),dtype=torch.float64,requires_grad=True);bias=torch.zeros_like(t,requires_grad=True)
    optimizer=torch.optim.LBFGS([t,bias],max_iter=80,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();scale=F.softplus(t)
        loss=.5*(F.binary_cross_entropy_with_logits(scale*co+bias,torch.zeros_like(co))+
                  F.binary_cross_entropy_with_logits(scale*cg+bias,torch.ones_like(cg)))+.001*(scale-1)**2
        loss.backward();return loss
    optimizer.step(closure)
    with torch.no_grad():
        zo=(F.softplus(t)*logits(net,o,bank)+bias).numpy()
        zg=(F.softplus(t)*logits(net,g.reshape(-1,12,d),bank)+bias).reshape(len(o),-1).numpy()
    q=sigmoid(zo);qg=sigmoid(zg)
    scores=np.c_[zo,q-qg.mean(1)]
    checkpoint=dict(variant=variant,d=d,state=state,mean=torch.tensor(mean),sd=torch.tensor(sd),
                    bank=bank,scale=F.softplus(t).detach(),bias=bias.detach())
    diagnostic=dict(origin_auc=auc(qg[240:320].ravel(),q[240:320]),
                    balanced_accuracy=.5*((q[240:320]<.5).mean()+(qg[240:320]>=.5).mean()),
                    parameters=sum(p.numel() for p in net.parameters()))
    return scores,checkpoint,diagnostic


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-50,50)))


def train_checkpoints(seed,role,pools,out):
    paths=[out/f'lm_{seed}_e{e}.pt' for e in BUDGETS]
    if all(p.exists() for p in paths):return
    torch.manual_seed(seed);net=rec.SentenceLM()
    bx,_=rec.encode(pools['background']);x,_=rec.encode(pools[role])
    rec.train(net,bx,np.ones(128,dtype=int),10)
    base=np.random.default_rng(81231).permutation(np.repeat(np.arange(4),64))
    repeats=rec.LEVELS[(base+(seed%1000-1))%4]
    indices=torch.repeat_interleave(torch.arange(256),torch.tensor(repeats))
    opt=torch.optim.Adam(net.parameters(),lr=.003);history=[]
    for epoch in range(1,max(BUDGETS)+1):
        net.train();values=[]
        for ids in torch.randperm(len(indices)).split(32):
            b=x[indices[ids]];loss=F.cross_entropy(net(b[:,:-1]).reshape(-1,96),b[:,1:].reshape(-1),ignore_index=0)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();values.append(loss.item())
        history.append(float(np.mean(values)))
        if epoch in BUDGETS:
            torch.save(dict(state=net.state_dict(),repeats=torch.tensor(repeats),history=history,epochs=epoch,seed=seed),out/f'lm_{seed}_e{epoch}.pt')
            print(f'  LM {seed}: checkpoint {epoch} epochs, loss {history[-1]:.3f}',flush=True)


def features(seed,role,epoch,pools,args):
    path=args.output/f'features_{seed}_e{epoch}.npz'
    if path.exists():
        with np.load(path) as f:r={k:f[k] for k in f.files}
        assert int(r['samples'])==args.samples and str(r['version'])==VERSION
        return r
    checkpoint=torch.load(args.output/f'lm_{seed}_e{epoch}.pt',weights_only=True)
    net=rec.SentenceLM();net.load_state_dict(checkpoint['state']);net.eval()
    torch.manual_seed(seed+epoch*10000)
    raw,_=rec.recover(net,pools['auxiliary']+pools[role],args.samples,return_embeddings=True)
    r=dict(original=raw['original_trajectory'],generated=raw['generated_trajectories'],
           loss=raw['teacher_full_loss'],teacher=raw['teacher_ce'],
           rollout=raw['rollout_ce'].transpose(0,2,1),
           repeats=checkpoint['repeats'].numpy(),samples=np.array(args.samples),version=np.array(VERSION))
    statistics=[raw[k][:,:,1:].mean(2) for k in ['suffix_exact','suffix_accuracy','tail_exact','tail_accuracy']]
    r['recovery_stats']=np.concatenate(statistics+[raw['suffix_exact'][:,:,0]],axis=1)
    np.savez_compressed(path,**r);return r


def model_scores(seed,epoch,r,args):
    path=args.output/f'scores_{seed}_e{epoch}.npz'
    if path.exists():
        with np.load(path) as f:return {k:f[k] for k in f.files}
    result=dict(loss=r['loss'][320:],labels=(r['repeats']>0).astype(float),repeats=r['repeats'],
                loss_curves=np.c_[r['teacher'][320:],r['rollout'][320:,1:].mean(1)],
                recovery_stats=r['recovery_stats'][320:])
    diagnostics=[]
    for j,v in enumerate(VARIANTS):
        if v=='loss_origin':o=r['teacher'][:,:,None];g=r['rollout'][:,:,:,None]
        else:o=r['original'];g=r['generated']
        score,checkpoint,diagnostic=fit_origin(o,g,v,seed+epoch*100+j)
        result[v]=score[320:];diagnostics.append(dict(seed=seed,epochs=epoch,variant=v,**diagnostic))
        torch.save(checkpoint,args.output/f'origin_{seed}_e{epoch}_{v}.pt')
    (args.output/f'origin_metrics_{seed}_e{epoch}.json').write_text(json.dumps(diagnostics,indent=2))
    np.savez_compressed(path,**result);return result


def ridge_logistic(x,y,penalty):
    x=torch.tensor(x,dtype=torch.float64);y=torch.tensor(y,dtype=torch.float64)
    w=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True);b=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([w,b],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=F.binary_cross_entropy_with_logits(x@w+b,y)+penalty*w.square().sum()/2
        loss.backward();return loss
    opt.step(closure);return w.detach().numpy(),float(b.detach())


def loss_basis(loss,mean,sd,knots):
    z=(loss-mean)/sd
    return np.c_[z,np.maximum(z[:,None]-knots[None,:],0)]


def anchor_geometry(vectors,anchors):
    """Common semantic anchor coordinates, invariant to neuron rotations/permutations."""
    center=anchors.mean(0);a=anchors-center;x=vectors-center
    a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8)
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    return x@a.T


def probe_design(r,name,loss_parameters):
    base=loss_basis(r['loss'],*loss_parameters)
    mapping={'LOSS + CNN':'cnn','LOSS + flat':'flat','LOSS + shuffle':'shuffle',
             'LOSS + pooled':'pooled','LOSS + loss-origin':'loss_origin',
             'LOSS + loss curves':'loss_curves','LOSS + recovery':'recovery_stats'}
    if name=='LOSS':return base
    if name==EXTRA_PROBE:return np.c_[base,r['cnn_representation']]
    if name==ANCHOR_PROBE:return np.c_[base,r['cnn_anchor_representation']]
    if name in mapping:return np.c_[base,r[mapping[name]]]
    extra=np.c_[r['loss_curves'],r['recovery_stats']]
    if name=='LOSS + statistics + CNN':extra=np.c_[extra,r['cnn']]
    return np.c_[base,extra]


def nll(p,y):
    p=np.clip(p,1e-9,1-1e-9)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def conditional_probes(records,epoch,out):
    loss=np.concatenate([records[s]['loss'] for s in SEEDS['train']]);mean=loss.mean();sd=max(loss.std(),1e-5)
    knots=np.quantile((loss-mean)/sd,np.arange(1,10)/10);lp=(mean,sd,knots)
    predictions={};fitted={}
    for name in PROBES+[EXTRA_PROBE,ANCHOR_PROBE]:
        x=np.concatenate([probe_design(records[s],name,lp) for s in SEEDS['train']])
        y=np.concatenate([records[s]['labels'] for s in SEEDS['train']])
        vx=np.concatenate([probe_design(records[s],name,lp) for s in SEEDS['validation']])
        vy=np.concatenate([records[s]['labels'] for s in SEEDS['validation']])
        mu=x.mean(0);scale=np.maximum(x.std(0),1e-4);x=(x-mu)/scale;vx=(vx-mu)/scale
        candidates=[]
        for penalty in LAMBDAS:
            w,b=ridge_logistic(x,y,penalty);value=nll(sigmoid(vx@w+b),vy).mean()
            candidates.append((float(value),penalty,w,b))
        value,penalty,w,b=min(candidates,key=lambda a:a[0])
        ps=np.stack([sigmoid(((probe_design(records[s],name,lp)-mu)/scale)@w+b) for s in SEEDS['target']])
        predictions[name]=ps
        fitted[name]=dict(validation_nll=value,penalty=penalty,mean=mu.tolist(),scale=scale.tolist(),
                          weights=w.tolist(),bias=b,loss_mean=float(mean),loss_sd=float(sd),knots=knots.tolist())
    # Additional conventional comparator: fixed negative-NLL ranking. This
    # guards against an apparent gain caused by a poorly generalized spline.
    rawx=-(loss-mean)/sd
    a,b=calibrate(rawx,np.concatenate([records[s]['labels'] for s in SEEDS['train']]))
    predictions['Raw LOSS']=np.stack([sigmoid(a*(-(records[s]['loss']-mean)/sd)+b) for s in SEEDS['target']])
    fitted['Raw LOSS']=dict(scale=a,bias=b,loss_mean=float(mean),loss_sd=float(sd),penalty=0)
    (out/f'probes_e{epoch}.json').write_text(json.dumps(fitted,indent=2))
    np.savez_compressed(out/f'predictions_e{epoch}.npz',**predictions)
    labels=np.stack([records[s]['labels'] for s in SEEDS['target']])
    rng=np.random.default_rng(19293)
    # Two-way bootstrap: shared sentence identities and fitted target models.
    draws=[(rng.integers(4,size=4),rng.integers(256,size=256)) for _ in range(1000)]
    rows=[];boots={}
    for name,ps in predictions.items():
        point=np.mean([auc(p[y==1],p[y==0]) for p,y in zip(ps,labels)])
        bauc=[];bnll=[]
        for models,ids in draws:
            bauc.append(np.mean([auc(ps[i,ids][labels[i,ids]==1],ps[i,ids][labels[i,ids]==0]) for i in models]))
            bnll.append(np.mean([nll(ps[i,ids],labels[i,ids]).mean() for i in models]))
        boots[name]=(np.array(bauc),np.array(bnll))
        lo,hi=np.quantile(bauc,[.025,.975])
        rows.append(dict(epochs=epoch,method=name,auc=point,ci_low=lo,ci_high=hi,
                         nll=nll(ps,labels).mean(),brier=np.mean((ps-labels)**2),penalty=fitted[name]['penalty']))
    paired=[];base=next(r for r in rows if r['method']=='LOSS')
    for name in PROBES[1:]+[EXTRA_PROBE,ANCHOR_PROBE]:
        row=next(r for r in rows if r['method']==name)
        da=boots[name][0]-boots['LOSS'][0];dn=boots['LOSS'][1]-boots[name][1]
        alo,ahi=np.quantile(da,[.025,.975]);nlo,nhi=np.quantile(dn,[.025,.975])
        paired.append(dict(epochs=epoch,comparison=name+' minus LOSS',delta_auc=row['auc']-base['auc'],
                           auc_low=alo,auc_high=ahi,nll_improvement=base['nll']-row['nll'],nll_low=nlo,nll_high=nhi))
    # Stronger control: does CNN add anything over BOTH loss and recovery curves?
    left,right='LOSS + statistics + CNN','LOSS + statistics'
    da=boots[left][0]-boots[right][0];lo,hi=np.quantile(da,[.025,.975])
    l,r=[next(x for x in rows if x['method']==n) for n in [left,right]]
    dn=boots[right][1]-boots[left][1];nl,nh=np.quantile(dn,[.025,.975])
    paired.append(dict(epochs=epoch,comparison='CNN added to statistics',delta_auc=l['auc']-r['auc'],auc_low=lo,auc_high=hi,
                       nll_improvement=r['nll']-l['nll'],nll_low=nl,nll_high=nh))
    for right in ['LOSS + flat','LOSS + shuffle','LOSS + pooled']:
        left='LOSS + CNN';da=boots[left][0]-boots[right][0];lo,hi=np.quantile(da,[.025,.975])
        l,r=[next(x for x in rows if x['method']==n) for n in [left,right]]
        dn=boots[right][1]-boots[left][1];nl,nh=np.quantile(dn,[.025,.975])
        paired.append(dict(epochs=epoch,comparison='CNN versus '+right.removeprefix('LOSS + '),delta_auc=l['auc']-r['auc'],
            auc_low=lo,auc_high=hi,nll_improvement=r['nll']-l['nll'],nll_low=nl,nll_high=nh))
    for name in ['LOSS','LOSS + CNN','LOSS + statistics + CNN']:
        left=next(x for x in rows if x['method']==name);right=next(x for x in rows if x['method']=='Raw LOSS')
        da=boots[name][0]-boots['Raw LOSS'][0];lo,hi=np.quantile(da,[.025,.975])
        dn=boots['Raw LOSS'][1]-boots[name][1];nl,nh=np.quantile(dn,[.025,.975])
        paired.append(dict(epochs=epoch,comparison=name+' minus Raw LOSS',delta_auc=left['auc']-right['auc'],
            auc_low=lo,auc_high=hi,nll_improvement=right['nll']-left['nll'],nll_low=nl,nll_high=nh))
    return rows,paired


def loss_match(loss,positive,negative,caliper):
    """Greedy globally closest disjoint pairs, using loss and labels ONLY."""
    p=np.flatnonzero(positive);n=np.flatnonzero(negative)
    distances=np.abs(loss[p,None]-loss[n][None,:]);flat=np.argsort(distances.ravel(),kind='stable')
    used_p=set();used_n=set();pairs=[]
    for idx in flat:
        i,j=np.unravel_index(idx,distances.shape)
        if distances[i,j]>caliper:break
        if i in used_p or j in used_n:continue
        used_p.add(i);used_n.add(j);pairs.append((int(p[i]),int(n[j])))
    return np.array(pairs,dtype=int).reshape(-1,2)


def matched_analysis(records,epoch,out):
    rows=[];raw={}
    for seed in SEEDS['target']:
        r=records[seed]
        for task,positive,negative in [('seen_vs_unseen',r['repeats']>0,r['repeats']==0),
                                       ('high_vs_low_exposure',r['repeats']==16,r['repeats']==1)]:
            for caliper in CALIPERS:
                pairs=loss_match(r['loss'],positive,negative,caliper)
                if len(pairs):
                    a,b=pairs.T;raw[f'{seed}_{task}_{caliper}_pairs']=pairs
                    raw[f'{seed}_{task}_{caliper}_loss_gap']=r['loss'][a]-r['loss'][b]
                    for variant in VARIANTS:
                        for j,score in enumerate(['class1_logit','paired_gap']):
                            delta=r[variant][a,j]-r[variant][b,j]
                            raw[f'{seed}_{task}_{caliper}_{variant}_{score}']=delta
                for variant in VARIANTS:
                    for j,score in enumerate(['class1_logit','paired_gap']):
                        if len(pairs):
                            delta=r[variant][a,j]-r[variant][b,j]
                            loss_delta=r['loss'][b]-r['loss'][a]
                            win=np.mean((delta>0)+.5*(delta==0));loss_win=np.mean((loss_delta>0)+.5*(loss_delta==0))
                            gap=np.mean(np.abs(r['loss'][a]-r['loss'][b]));auc_loss=auc(-r['loss'][a],-r['loss'][b])
                            members=r['repeats'][a];distribution=','.join(str(int((members==l).sum())) for l in [1,4,16])
                        else:win=loss_win=gap=auc_loss='';distribution='0,0,0'
                        rows.append(dict(epochs=epoch,seed=seed,task=task,caliper=caliper,variant=variant,score=score,
                            pairs=len(pairs),score_winrate=win,loss_winrate=loss_win,mean_abs_loss_gap=gap,
                            matched_loss_auc=auc_loss,positive_exposure_mix=distribution))
    np.savez_compressed(out/f'matched_pairs_e{epoch}.npz',**raw)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/membrain_beyond_loss')
    p.add_argument('--samples',type=int,default=4);p.add_argument('--analysis-only',action='store_true');args=p.parse_args()
    assert args.samples>=4;torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    pools,excluded=fresh_pools()
    plan=dict(version=VERSION,budgets=BUDGETS,seeds=SEEDS,samples=args.samples,origin_variants=VARIANTS,
              conditional_probes=PROBES,match_calipers=CALIPERS,primary_caliper=.02,probe_penalties=LAMBDAS,
              source_sha256=rec.previous.SOURCE_SHA,excluded_previous_sentences=excluded,
              role_pool_sizes={k:len(v) for k,v in pools.items()},
              origin_membership_labels=False,conditional_probe_membership_labels=True,
              primary_budget_epochs=5,
              primary_test='held-out target increment over flexible whole-sentence LOSS',
              uncertainty='two-way target-model and shared-sentence bootstrap, 1000 replicates')
    plan_path=args.output/'analysis_plan.json'
    if plan_path.exists():
        stored=json.loads(plan_path.read_text())
        assert all(plan.get(k)==v for k,v in stored.items()),'Use fresh output for changed plan.'
    else:plan_path.write_text(json.dumps(plan,indent=2))
    effective=dict(plan,additional_comparator='Raw LOSS',exploratory_probes=[EXTRA_PROBE,ANCHOR_PROBE],
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'effective_analysis_config.json').write_text(json.dumps(effective,indent=2))
    related=Path(__file__).with_name('membrain_related_work.json')
    if related.exists():(args.output/'related_work_verification.json').write_text(related.read_text())
    (args.output/'sentence_splits.json').write_text(json.dumps(pools,indent=2))
    for role,seeds in SEEDS.items():
        for seed in seeds:
            if args.analysis_only:continue
            print(f'Train fresh {role} LM {seed}',flush=True);train_checkpoints(seed,role,pools,args.output)
            for epoch in BUDGETS:
                print(f'  Extract/fit origin networks {seed}, epoch {epoch}',flush=True)
                r=features(seed,role,epoch,pools,args);model_scores(seed,epoch,r,args)
    metrics=[];paired=[];matched=[]
    for epoch in BUDGETS:
        records={}
        for seeds in SEEDS.values():
            for seed in seeds:
                with np.load(args.output/f'scores_{seed}_e{epoch}.npz') as f:records[seed]={k:f[k] for k in f.files}
                if 'cnn_anchor_representation' not in records[seed]:
                    checkpoint=torch.load(args.output/f'origin_{seed}_e{epoch}_cnn.pt',weights_only=True)
                    net=Network('cnn');net.load_state_dict(checkpoint['state']);net.eval()
                    with np.load(args.output/f'features_{seed}_e{epoch}.npz') as f:
                        x=torch.tensor((f['original']-checkpoint['mean'].numpy())/checkpoint['sd'].numpy(),dtype=torch.float32)
                        ids=np.linspace(0,159,16).astype(int)
                        g=torch.tensor((f['generated'][ids,1]-checkpoint['mean'].numpy())/checkpoint['sd'].numpy(),dtype=torch.float32)
                    with torch.no_grad():
                        latent=net.head[:-1](net.body(x.transpose(1,2))).numpy()
                        glatent=net.head[:-1](net.body(g.transpose(1,2))).numpy()
                    records[seed]['cnn_representation']=latent[320:]
                    records[seed]['cnn_anchor_representation']=anchor_geometry(latent[320:],np.r_[latent[ids],glatent])
                    np.savez_compressed(args.output/f'scores_{seed}_e{epoch}.npz',**records[seed])
        rows,comparisons=conditional_probes(records,epoch,args.output)
        metrics+=rows;paired+=comparisons;matched+=matched_analysis(records,epoch,args.output)
        print(f'Completed held-out analysis at {epoch} epochs',flush=True)
        for name,data in [('conditional_metrics',metrics),('paired_increments',paired),('loss_matched',matched)]:
            with (args.output/f'{name}.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print('DONE',flush=True)


if __name__=='__main__':main()
