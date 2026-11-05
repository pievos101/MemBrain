"""Membrain: free-running predictions scored against the TRUE continuation.

This is a separate experiment. Earlier generated self-likelihood caches are not
compatible with this definition and are never read here.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import membrain_loss_curves as prev
import membrain_geometry as mg

SCORING='free_running_true_target_cross_entropy_v1'
METHODS=['Membrain auxiliary curve','Membrain paired curves','Membrain rollout curve',
         'Membrain rollout mean','Membrain excess mean','Original curve MLP',
         'LOSS same windows','LOSS full text','Min-K 20%']
MLP_METHODS=['Membrain paired curves','Membrain rollout curve','Original curve MLP']
COMPARISONS=[('Membrain auxiliary curve','LOSS full text'),
             ('Membrain paired curves','LOSS full text'),
             ('Membrain paired curves','LOSS same windows'),
             ('Membrain paired curves','Original curve MLP'),
             ('Membrain rollout curve','Original curve MLP'),
             ('Membrain rollout mean','LOSS same windows')]


@torch.no_grad()
def free_running_losses(lm,batch,p,samples,horizon=8,sampler=None):
    """CE target is always the true token; recurrent input is always sampled.

    Returned shape: documents, sampled paths, horizon. True suffix tokens are
    NEVER passed into the GRU after prefix initialization.
    """
    lm.eval()
    _,state=lm.gru(lm.embedding(batch[:,:p].repeat_interleave(samples,0)))
    losses=[]
    for j in range(horizon):
        logp=lm.output(state[-1]).log_softmax(-1)
        true_token=batch[:,p+j].repeat_interleave(samples)
        losses.append(-logp.gather(1,true_token[:,None])[:,0])
        sampled=(torch.multinomial(logp.exp(),1) if sampler is None else sampler(logp.exp()))
        _,state=lm.gru(lm.embedding(sampled),state)
    return torch.stack(losses,1).reshape(len(batch),samples,horizon)


@torch.no_grad()
def curves(lm,docs,samples):
    teacher=[];rollout=[];token_losses=[];first_errors=[]
    lm.eval()
    for batch in docs.split(160):
        nll=torch.nn.functional.cross_entropy(lm(batch[:,:-1]).transpose(1,2),batch[:,1:],reduction='none')
        token_losses.append(nll.numpy())
        teacher.append(torch.stack([nll[:,p-1:p+7].mean(1) for p in prev.PREFIXES],1).numpy())
        rows=[]
        for p in prev.PREFIXES:
            values=free_running_losses(lm,batch,p,samples)
            first_errors.append(float((values[:,:,0]-nll[:,p-1,None]).abs().max()))
            rows.append(values.mean(2))
        rollout.append(torch.stack(rows,2).numpy())
    # At the first suffix token the two histories are identical.
    assert max(first_errors)<1e-5,first_errors
    return np.concatenate(teacher),np.concatenate(rollout),np.concatenate(token_losses),max(first_errors)


def model_run(seed,scenario,samples,path):
    cfg=prev.SCENARIOS[scenario];torch.manual_seed(seed)
    docs,difficulty=prev.corpus(seed,cfg['mixed'],cfg.get('natural',False))
    lm=prev.base.LM()
    history=prev.train_lm(lm,docs,cfg['epochs'],cfg['repeat'])
    torch.save(lm.state_dict(),path.with_suffix('.pt'))
    torch.manual_seed(seed+10000)
    teacher,rollout,tokens,error=curves(lm,docs,samples)
    # Auxiliary classifier distinguishes teacher-forced and free-running loss
    # curves on the SAME auxiliary true continuations, without membership labels.
    gap,acc=prev.auxiliary_curve_score(teacher,rollout)
    ids=prev.EVAL_IDS
    return dict(teacher=teacher[ids],rollout=rollout[ids],token_losses=tokens[ids],
                auxiliary_gap=gap[ids],aux_accuracy=np.array(acc),history=history,
                difficulty=difficulty[ids],first_step_error=np.array(error),
                scoring=np.array(SCORING),samples=np.array(samples),seed=np.array(seed))


def design(r,method):
    teacher,rollout=r['teacher'],r['rollout'].mean(1)
    if method=='Membrain auxiliary curve':return r['auxiliary_gap'][:,None]
    if method=='Membrain paired curves':return np.c_[teacher,rollout]
    if method=='Membrain rollout curve':return rollout
    if method=='Membrain rollout mean':return -rollout.mean(1,keepdims=True)
    if method=='Membrain excess mean':return (rollout-teacher).mean(1,keepdims=True)
    if method=='Original curve MLP':return teacher
    if method=='LOSS same windows':return -teacher.mean(1,keepdims=True)
    if method=='LOSS full text':return -r['token_losses'].mean(1,keepdims=True)
    if method=='Min-K 20%':return -np.sort(r['token_losses'],axis=1)[:,-10:].mean(1,keepdims=True)
    raise ValueError(method)


def evaluate(records,scenario,out):
    predictions={};params={}
    for index,method in enumerate(METHODS):
        torch.manual_seed(88000+index)
        if method in MLP_METHODS:
            x=np.concatenate([design(records[s],method) for s in prev.TRAIN_SEEDS[:3]])
            predict,saved=prev.fit_mlp(x,np.tile(prev.LABELS,3),design(records[prev.TRAIN_SEEDS[3]],method),prev.LABELS)
            raw=lambda r:predict(design(r,method))
        else:
            x=np.concatenate([design(records[s],method) for s in prev.TRAIN_SEEDS])
            mean,sd=x.mean(0),np.maximum(x.std(0),1e-5)
            if method in ['LOSS same windows','LOSS full text','Min-K 20%','Membrain rollout mean']:
                scale,b=mg.calibrate(((x-mean)/sd)[:,0],np.tile(prev.LABELS,4));w=np.array([scale])
            else:w,b=mg.logistic((x-mean)/sd,np.tile(prev.LABELS,4))
            raw=lambda r:((design(r,method)-mean)/sd)@w+b
            saved=dict(mean=mean.tolist(),sd=sd.tolist(),weights=w.tolist(),bias=b)
        cx=np.concatenate([raw(records[s]) for s in prev.CAL_SEEDS])
        scale,bias=mg.calibrate(cx,np.tile(prev.LABELS,2))
        predictions[method]=np.stack([mg.sigmoid(scale*raw(records[s])+bias) for s in prev.TEST_SEEDS])
        params[method]=dict(learner=saved,calibration_scale=scale,calibration_bias=bias)
    np.savez_compressed(out/f'predictions_{scenario}.npz',**predictions)
    (out/f'learners_{scenario}.json').write_text(json.dumps(params,indent=2))
    rows=[];contrasts=[];rng=np.random.default_rng(8732)
    for group,positive in [('all',np.arange(240)),('ordinary',np.arange(200)),('strong',np.arange(200,240))]:
        draws=[[(int(i),rng.choice(positive,len(positive)),rng.integers(240,640,400))
                 for i in rng.integers(5,size=5)] for _ in range(1000)]
        boots={};means={}
        for method,ps in predictions.items():
            value=np.mean([prev.auc(p[positive],p[240:]) for p in ps]);means[method]=value
            bs=np.array([np.mean([prev.auc(ps[i,p],ps[i,n]) for i,p,n in draw]) for draw in draws]);boots[method]=bs
            lo,hi=np.quantile(bs,[.025,.975])
            tpr=np.mean([np.mean(p[positive]>np.quantile(p[240:],.95,method='higher')) for p in ps])
            rows.append(dict(scenario=scenario,group=group,method=method,auc=value,ci_low=lo,ci_high=hi,
                tpr_at_5pct_fpr=tpr,brier=np.mean((ps-prev.LABELS)**2) if group=='all' else '',
                ece=mg.ece(ps.ravel(),np.tile(prev.LABELS,5)) if group=='all' else ''))
        for left,right in COMPARISONS:
            lo,hi=np.quantile(boots[left]-boots[right],[.025,.975])
            contrasts.append(dict(scenario=scenario,group=group,comparison=f'{left} minus {right}',
                                  delta_auc=means[left]-means[right],ci_low=lo,ci_high=hi))
    return rows,contrasts


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/membrain_rollout'))
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--scenarios',nargs='+',choices=list(prev.SCENARIOS),default=list(prev.SCENARIOS));args=p.parse_args()
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    def retained(name):
        path=args.output/f'{name}.csv'
        if not path.exists():return []
        with path.open() as f:return [r for r in csv.DictReader(f) if r['scenario'] not in args.scenarios]
    metrics=retained('metrics');comparisons=retained('paired_comparisons')
    for scenario in args.scenarios:
        records={}
        for seed in prev.TRAIN_SEEDS+prev.CAL_SEEDS+prev.TEST_SEEDS:
            path=args.output/f'model_{scenario}_{seed}.npz'
            if path.exists():
                with np.load(path) as f:records[seed]={k:f[k] for k in f.files}
                assert str(records[seed]['scoring'])==SCORING
                assert int(records[seed]['samples'])==args.samples
            else:
                print(f'Membrain TRUE-target rollout: {scenario} seed={seed}',flush=True)
                records[seed]=model_run(seed,scenario,args.samples,path)
                np.savez_compressed(path,**records[seed])
        rows,paired=evaluate(records,scenario,args.output);metrics+=rows;comparisons+=paired
        for name,items in [('metrics',metrics),('paired_comparisons',comparisons)]:
            with (args.output/f'{name}.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
        print(f'Completed {scenario}',flush=True)
    (args.output/'config.json').write_text(json.dumps(dict(scoring=SCORING,scenarios=prev.SCENARIOS,
        samples=args.samples,prefixes=prev.PREFIXES,horizon=8,temperature=1.,
        train_seeds=prev.TRAIN_SEEDS,calibration_seeds=prev.CAL_SEEDS,target_seeds=prev.TEST_SEEDS,
        membership_prior=float(prev.LABELS.mean()),source_sha256=prev.SOURCE_SHA,
        torch=torch.__version__,numpy=np.__version__),indent=2))
    print('DONE',flush=True)


if __name__=='__main__':main()
