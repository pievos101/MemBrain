"""Membrain scalar continuation-loss curves versus information-matched LOSS controls."""
import argparse
import copy
import csv
import hashlib
import json
import os
import string
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import corrected_benchmark as base
import membrain_geometry as mg

PREFIXES=[8,16,24,32]
HORIZON=8
SCENARIOS={'null':dict(epochs=0,repeat=0,mixed=False),
           'early':dict(epochs=3,repeat=4,mixed=False),
           'mixed_difficulty':dict(epochs=3,repeat=4,mixed=True),
           'overfit_control':dict(epochs=30,repeat=15,mixed=False),
           'natural_text':dict(epochs=10,repeat=4,mixed=False,natural=True)}
SOURCE=Path(__file__).resolve().parents[1]/'results/membrain_loss_curves/source/tinyshakespeare.txt'
SOURCE_SHA='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
TRAIN_SEEDS=list(range(41001,41005))
CAL_SEEDS=list(range(42001,42003))
TEST_SEEDS=list(range(43001,43006))
EVAL_IDS=np.r_[np.arange(200),np.arange(400,440),np.arange(440,840)]
LABELS=np.r_[np.ones(240),np.zeros(400)]
METHODS=['Membrain auxiliary curve','Membrain paired curves','Original curve MLP',
         'Generated-reference gap','LOSS same windows','LOSS full text','Min-K 20%']


def auc(positive,negative):
    """Exact tie-aware AUC via sorted negatives, avoiding a large pair matrix."""
    n=np.sort(negative)
    return float(np.mean((np.searchsorted(n,positive,'left')+np.searchsorted(n,positive,'right'))/2)/len(n))


def corpus(seed,mixed,natural=False):
    rng=np.random.default_rng(seed)
    if natural:
        data=SOURCE.read_bytes()
        assert hashlib.sha256(data).hexdigest()==SOURCE_SHA,'Source text checksum mismatch'
        alphabet=string.ascii_letters+' '
        mapping={c:i for i,c in enumerate(alphabet)}
        text=data.decode('utf-8')
        tokens=[mapping.get(c,52) for c in text]
        chunks=[];seen=set()
        # Nonoverlapping raw-text blocks, no sliding windows across splits.
        for start in range(0,len(tokens)-47,48):
            row=tuple(tokens[start:start+48])
            if row not in seen:seen.add(row);chunks.append(row)
        chosen=rng.choice(len(chunks),1120,replace=False)
        return torch.tensor(np.array(chunks)[chosen]),np.full(1120,-1)
    matrices=[rng.dirichlet(np.ones(53)*a,size=53) for a in ([.02,1.] if mixed else [.12])]
    sizes=[400,40,400,200,80]
    docs=[]; difficulty=[]; seen=set()
    # Same source(s), with identical difficulty proportions in every split.
    for size in sizes:
        types=np.tile(np.arange(len(matrices)),size//len(matrices))
        rng.shuffle(types)
        for t in types:
            while True:
                x=[int(rng.integers(53))]
                for _ in range(47): x.append(int(rng.choice(53,p=matrices[t][x[-1]])))
                if tuple(x) not in seen: break
            seen.add(tuple(x)); docs.append(x); difficulty.append(int(t))
    assert len(seen)==1120
    return torch.tensor(docs),np.array(difficulty)


def train_lm(lm,docs,epochs,repeat):
    data=torch.cat([docs[:400],docs[400:440].repeat(repeat,1)])
    opt=torch.optim.Adam(lm.parameters(),lr=.003)
    history=[]
    for _ in range(epochs):
        values=[]
        for ids in torch.randperm(len(data)).split(32):
            b=data[ids]; loss=F.cross_entropy(lm(b[:,:-1]).reshape(-1,53),b[:,1:].reshape(-1))
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(lm.parameters(),1.);opt.step()
            values.append(loss.item())
        history.append(np.mean(values))
    return np.array(history)


@torch.no_grad()
def loss_curves(lm,docs,samples):
    """Exactly aligned next-token loss, raw LM probabilities; temperature=1."""
    lm.eval(); originals=[]; generations=[]; token_losses=[]
    for batch in docs.split(160):
        nll=F.cross_entropy(lm(batch[:,:-1]).transpose(1,2),batch[:,1:],reduction='none')
        token_losses.append(nll.numpy())
        originals.append(torch.stack([nll[:,p-1:p+HORIZON-1].mean(1) for p in PREFIXES],1).numpy())
        curves=[]
        for p in PREFIXES:
            prefix=batch[:,:p].repeat_interleave(samples,0)
            _,state=lm.gru(lm.embedding(prefix)); losses=[]
            for _ in range(HORIZON):
                logits=lm.output(state[-1]); logp=logits.log_softmax(-1)
                token=torch.multinomial(logp.exp(),1)
                losses.append(-logp.gather(1,token)[:,0])
                _,state=lm.gru(lm.embedding(token),state)
            curves.append(torch.stack(losses,1).mean(1).reshape(len(batch),samples))
        generations.append(torch.stack(curves,2).numpy())
    return np.concatenate(originals),np.concatenate(generations),np.concatenate(token_losses)


def fit_mlp(train,y,val,vy):
    mean=train.mean(0); sd=np.maximum(train.std(0),1e-5)
    x=torch.tensor((train-mean)/sd,dtype=torch.float32)
    vx=torch.tensor((val-mean)/sd,dtype=torch.float32)
    y=torch.tensor(y,dtype=torch.float32);vy=torch.tensor(vy,dtype=torch.float32)
    net=nn.Sequential(nn.Linear(x.shape[1],16),nn.Tanh(),nn.Linear(16,1))
    opt=torch.optim.Adam(net.parameters(),lr=.003,weight_decay=.01)
    best,saved,stale=float('inf'),None,0
    for _ in range(160):
        for ids in torch.randperm(len(x)).split(128):
            loss=F.binary_cross_entropy_with_logits(net(x[ids])[:,0],y[ids])
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():value=F.binary_cross_entropy_with_logits(net(vx)[:,0],vy).item()
        if value<best-1e-5:best,saved,stale=value,copy.deepcopy(net.state_dict()),0
        else:stale+=1
        if stale>=15:break
    net.load_state_dict(saved);net.eval()
    def predict(data):
        with torch.no_grad():return net(torch.tensor((data-mean)/sd,dtype=torch.float32))[:,0].numpy()
    return predict,dict(mean=mean.tolist(),sd=sd.tolist(),
                        state={k:v.tolist() for k,v in saved.items()})


def auxiliary_curve_score(o,g):
    # Original/generated supervision only, independent auxiliary documents.
    x=np.r_[o[840:1040],g[840:1040,0]]
    y=np.r_[np.zeros(200),np.ones(200)]
    vx=np.r_[o[1040:],g[1040:,0]];vy=np.r_[np.zeros(80),np.ones(80)]
    predict,_=fit_mlp(x,y,vx,vy)
    lo=predict(o);lg=predict(g.reshape(-1,4)).reshape(len(o),-1).mean(1)
    accuracy=np.mean((predict(vx)>0)==vy)
    return lo-lg,accuracy


def run_model(seed,scenario,samples):
    cfg=SCENARIOS[scenario];torch.manual_seed(seed)
    docs,difficulty=corpus(seed,cfg['mixed'],cfg.get('natural',False));lm=base.LM()
    history=train_lm(lm,docs,cfg['epochs'],cfg['repeat'])
    torch.manual_seed(seed+10000)
    o,g,tokens=loss_curves(lm,docs,samples)
    gap,acc=auxiliary_curve_score(o,g)
    return dict(original=o[EVAL_IDS],generated=g[EVAL_IDS],token_losses=tokens[EVAL_IDS],
                auxiliary_gap=gap[EVAL_IDS],aux_accuracy=np.array(acc),history=history,
                difficulty=difficulty[EVAL_IDS],seed=np.array(seed),samples=np.array(samples))


def design(r,method):
    o,g=r['original'],r['generated'].mean(1)
    if method=='Membrain auxiliary curve':return r['auxiliary_gap'][:,None]
    if method=='Membrain paired curves':return np.c_[o,g]
    if method=='Original curve MLP':return o
    if method=='Generated-reference gap':return (g-o).mean(1,keepdims=True)
    if method=='LOSS same windows':return -o.mean(1,keepdims=True)
    if method=='LOSS full text':return -r['token_losses'].mean(1,keepdims=True)
    if method=='Min-K 20%':return -np.sort(r['token_losses'],axis=1)[:,-10:].mean(1,keepdims=True)
    raise ValueError(method)


def evaluate(records,scenario,out):
    predictions={};params={}
    for index,method in enumerate(METHODS):
        torch.manual_seed(88000+index)
        if method in ['Membrain paired curves','Original curve MLP']:
            # Three fitting shadows, fourth shadow for early stopping.
            train=np.concatenate([design(records[s],method) for s in TRAIN_SEEDS[:3]])
            predict,saved=fit_mlp(train,np.tile(LABELS,3),design(records[TRAIN_SEEDS[3]],method),LABELS)
            raw=lambda r:predict(design(r,method))
        else:
            x=np.concatenate([design(records[s],method) for s in TRAIN_SEEDS]); mean=x.mean(0);sd=np.maximum(x.std(0),1e-5)
            if method in ['LOSS same windows','LOSS full text','Min-K 20%']:
                # Conventional loss ranks lower NLL as more likely membership.
                # Calibrate that fixed direction; never reverse a LOSS baseline.
                scale,b=mg.calibrate(((x-mean)/sd)[:,0],np.tile(LABELS,4))
                w=np.array([scale])
            else:
                w,b=mg.logistic((x-mean)/sd,np.tile(LABELS,4))
            raw=lambda r:((design(r,method)-mean)/sd)@w+b
            saved=dict(mean=mean.tolist(),sd=sd.tolist(),weights=w.tolist(),bias=b)
        cx=np.concatenate([raw(records[s]) for s in CAL_SEEDS])
        scale,bias=mg.calibrate(cx,np.tile(LABELS,2))
        predictions[method]=np.stack([mg.sigmoid(scale*raw(records[s])+bias) for s in TEST_SEEDS])
        params[method]=dict(learner=saved,calibration_scale=scale,calibration_bias=bias)
    np.savez_compressed(out/f'predictions_{scenario}.npz',**predictions)
    (out/f'learners_{scenario}.json').write_text(json.dumps(params,indent=2))
    rows=[];contrasts=[]
    rng=np.random.default_rng(8732)
    # Reuse identical hierarchical observations across methods for paired contrasts.
    for group,positive in [('all',np.arange(240)),('ordinary',np.arange(200)),('strong',np.arange(200,240))]:
        draws=[[(int(i),rng.choice(positive,len(positive)),rng.integers(240,640,400))
                 for i in rng.integers(5,size=5)] for _ in range(1000)]
        boots={}
        for method,ps in predictions.items():
            values=[auc(p[positive],p[240:]) for p in ps]
            bs=np.array([np.mean([auc(ps[i,p],ps[i,n]) for i,p,n in draw]) for draw in draws]);boots[method]=bs
            lo,hi=np.quantile(bs,[.025,.975])
            tpr=[np.mean(p[positive]>np.quantile(p[240:],.95,method='higher')) for p in ps]
            rows.append(dict(scenario=scenario,group=group,method=method,auc=np.mean(values),ci_low=lo,ci_high=hi,
                             tpr_at_5pct_fpr=np.mean(tpr),brier=np.mean((ps-LABELS)**2) if group=='all' else '',
                             ece=mg.ece(ps.ravel(),np.tile(LABELS,5)) if group=='all' else ''))
        for left,right in [('Membrain auxiliary curve','LOSS full text'),('Membrain paired curves','LOSS full text'),
                           ('Membrain paired curves','LOSS same windows'),('Membrain paired curves','Original curve MLP'),
                           ('Generated-reference gap','LOSS same windows')]:
            delta=boots[left]-boots[right];lo,hi=np.quantile(delta,[.025,.975])
            means={r['method']:r['auc'] for r in rows if r['group']==group}
            contrasts.append(dict(scenario=scenario,group=group,comparison=f'{left} minus {right}',
                                  delta_auc=means[left]-means[right],ci_low=lo,ci_high=hi))
    return rows,contrasts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('results/membrain_loss_curves'))
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--scenarios',nargs='+',choices=list(SCENARIOS),default=list(SCENARIOS));args=parser.parse_args()
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    def retained(name):
        path=args.output/f'{name}.csv'
        if not path.exists():return []
        with path.open() as f:return [r for r in csv.DictReader(f) if r['scenario'] not in args.scenarios]
    allrows=retained('metrics');allcontrasts=retained('paired_comparisons')
    for scenario in args.scenarios:
        records={}
        for seed in TRAIN_SEEDS+CAL_SEEDS+TEST_SEEDS:
            path=args.output/f'model_{scenario}_{seed}.npz'
            if path.exists():
                with np.load(path) as f:records[seed]={k:f[k] for k in f.files}
                assert int(records[seed]['samples'])==args.samples,'Use a fresh output directory for changed settings.'
            else:
                print(f'Membrain loss curves: {scenario} seed={seed}',flush=True)
                records[seed]=run_model(seed,scenario,args.samples);np.savez_compressed(path,**records[seed])
        rows,contrasts=evaluate(records,scenario,args.output);allrows+=rows;allcontrasts+=contrasts
        print(f'Completed {scenario}',flush=True)
        # Save completed scenarios as checkpoints as well as at final completion.
        for name,items in [('metrics',allrows),('paired_comparisons',allcontrasts)]:
            with (args.output/f'{name}.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    (args.output/'config.json').write_text(json.dumps(dict(scenarios=SCENARIOS,samples=args.samples,
        prefixes=PREFIXES,horizon=HORIZON,temperature=1.,train_seeds=TRAIN_SEEDS,
        calibration_seeds=CAL_SEEDS,target_seeds=TEST_SEEDS,prior=float(LABELS.mean()),
        source_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        source_sha256=SOURCE_SHA,torch=torch.__version__,numpy=np.__version__),indent=2))
    print('DONE',flush=True)


if __name__=='__main__':main()
