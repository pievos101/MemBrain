"""Membrain: membership-supervised, probability-weighted continuation trajectories.

The feature extractor has no membership labels. It compares aligned TRUE tokens
with free-running generated tokens at increasing prefixes. A separate regularized
classifier learns membership from independent shadow models.
"""
import numpy as np
import torch
from torch.nn import functional as F


def token_weights(full_logp,short_logp):
    """Half uniform, half softmax of full-context gain over shortened context."""
    gain=np.clip(np.asarray(full_logp)-np.asarray(short_logp),-20,20)
    e=np.exp(gain-gain.max(axis=-1,keepdims=True))
    return .5/gain.shape[-1]+.5*e/e.sum(axis=-1,keepdims=True)


@torch.no_grad()
def recovery_features(model,sequences,details,fractions):
    """Return label-free prefix curves and old pooled hidden trajectories.

    SPW adaptive-prefix probability changes weight suffix positions. Hidden
    states are aligned AFTER each true/generated token, not averaged before
    comparison. Free-running CE always targets TRUE continuation tokens.
    """
    model.eval();x=torch.as_tensor(np.asarray(sequences),dtype=torch.long);n,length=x.shape
    if min(round(length*f) for f in fractions)<2:raise ValueError('Prefixes need >=2 tokens')
    gain=np.array([np.array(d['target_logp'])-np.array(d['shortened_logp']) for d in details])
    names=['pooled_cosine','aligned_cosine','weighted_cosine','accuracy','exact','rollout_ce',
           'rollout_excess_ce','weighted_rollout_ce','weighted_accuracy','teacher_ce']
    curves={k:[] for k in names};true=[];generated=[]
    for f in fractions:
        p=round(length*f);parts={k:[] for k in names};ht=[];hg=[]
        for start in range(0,n,64):
            b=x[start:start+64]
            tl,th,_=model(b,return_hidden=True)
            teacher_ce=F.cross_entropy(tl[:,p-1:length-1].transpose(1,2),b[:,p:],reduction='none')
            logits,_,cache=model(b[:,:p],return_hidden=True);hidden=[];tokens=[];ce=[]
            for j in range(p,length):
                ce.append(F.cross_entropy(logits[:,-1],b[:,j],reduction='none'))
                token=logits[:,-1].argmax(-1);tokens.append(token)
                logits,h,cache=model(token[:,None],cache=cache,return_hidden=True);hidden.append(h[:,0])
            gh=torch.stack(hidden,1);truth=th[:,p:];gen=torch.stack(tokens,1)
            aligned=F.cosine_similarity(truth,gh,dim=-1)
            losses=torch.stack(ce,1);match=(gen==b[:,p:]).float()
            w=torch.tensor(token_weights(gain[start:start+len(b),p-2:],np.zeros((len(b),length-p))),dtype=losses.dtype)
            a=truth.mean(1);g=gh.mean(1);ht.append(a.numpy());hg.append(g.numpy())
            values=dict(pooled_cosine=F.cosine_similarity(a,g),aligned_cosine=aligned.mean(1),
                weighted_cosine=(w*aligned).sum(1),accuracy=match.mean(1),exact=match.all(1).float(),
                rollout_ce=losses.mean(1),rollout_excess_ce=(losses-teacher_ce).mean(1),
                weighted_rollout_ce=(w*losses).sum(1),weighted_accuracy=(w*match).sum(1),
                teacher_ce=teacher_ce.mean(1))
            for k,v in values.items():parts[k].append(v.numpy())
        for k in names:curves[k].append(np.concatenate(parts[k]))
        true.append(np.concatenate(ht));generated.append(np.concatenate(hg))
    return {k:np.stack(v,1) for k,v in curves.items()},np.stack(true,1),np.stack(generated,1)


def assemble_features(curves,details,token_logp,fractions):
    """Explicit feature groups permit same-learner ablations of the contribution."""
    columns=[];names=[];groups=[]
    def add(name,group,a):
        columns.append(np.asarray(a));names.append(name);groups.append(group)
    for key,values in curves.items():
        group=('embedding' if 'cosine' in key else 'probability' if key=='teacher_ce' else 'generation')
        for j,f in enumerate(fractions):add(f'{key}@{f}',group,values[:,j])
        add(f'{key}:area',group,np.trapezoid(values,np.array(fractions),axis=1)/(fractions[-1]-fractions[0]))
        add(f'{key}:change',group,values[:,-1]-values[:,0])
        add(f'{key}:min',group,values.min(1));add(f'{key}:max',group,values.max(1))
    for key in ['spw','kl','weighted_similarity','weighted_max','mean_shortened_ratio','mean_initial_similarity']:
        add(key,'probability',[d[key] for d in details])
    add('LOSS','probability',token_logp.mean(1))
    k=max(1,int(np.ceil(token_logp.shape[1]*.2)))
    add('Min-k20%','probability',np.sort(token_logp,axis=1)[:,:k].mean(1))
    add('logp_sd','probability',token_logp.std(1))
    for q in [.1,.25,.5,.75,.9]:add(f'logp_quantile{q}','probability',np.quantile(token_logp,q,axis=1))
    gain=np.array([np.array(d['target_logp'])-np.array(d['shortened_logp']) for d in details])
    for key,v in [('gain_mean',gain.mean(1)),('gain_sd',gain.std(1)),('gain_max',gain.max(1))]:add(key,'probability',v)
    return np.stack(columns,1),names,groups


def feature_mask(groups,variant):
    groups=np.asarray(groups)
    if variant=='Membrain':return np.ones(len(groups),dtype=bool)
    if variant=='Probability-only learner':return groups=='probability'
    if variant=='Membrain without embeddings':return groups!='embedding'
    if variant=='Membrain curves only':return groups!='probability'
    raise ValueError(variant)


def fit_logistic(x,y,regularization):
    """Training-only normalization; no target statistics or target labels."""
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Nonfinite features')
    mu=x.mean(0);sd=np.maximum(x.std(0),1e-3)
    z=torch.tensor(np.clip((x-mu)/sd,-8,8));labels=torch.tensor(y)
    weight=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([weight,bias],max_iter=150,tolerance_grad=1e-9,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=F.binary_cross_entropy_with_logits(z@weight+bias,labels)+regularization*weight.square().sum()/2
        loss.backward();return loss
    opt.step(closure)
    return dict(mu=mu,sd=sd,weight=weight.detach().numpy(),bias=float(bias.detach()),regularization=regularization)


def decision(model,x):
    return np.clip((np.asarray(x)-model['mu'])/model['sd'],-8,8)@model['weight']+model['bias']


def calibrate(logits,y):
    """Positive-slope Platt calibration on completely separate shadow targets."""
    z=torch.tensor(logits,dtype=torch.float64);y=torch.tensor(y,dtype=torch.float64)
    log_slope=torch.zeros((),dtype=torch.float64,requires_grad=True)
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([log_slope,bias],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();slope=log_slope.clamp(-5,5).exp()
        loss=F.binary_cross_entropy_with_logits(slope*z+bias,y)+1e-5*(bias.square()+log_slope.square())
        loss.backward();return loss
    opt.step(closure)
    return dict(slope=float(log_slope.detach().clamp(-5,5).exp()),bias=float(bias.detach()))


def probabilities(calibration,logits):
    z=np.clip(calibration['slope']*np.asarray(logits)+calibration['bias'],-40,40)
    return 1/(1+np.exp(-z))


def fpr_threshold(nonmember_scores,fpr=.01):
    """Threshold for score>=threshold with <=floor(fpr*n) calibration false positives."""
    s=np.sort(np.asarray(nonmember_scores))[::-1];allowed=int(np.floor(fpr*len(s)))
    if not len(s) or not 0<=allowed<len(s):raise ValueError('Invalid FPR/count')
    return float(np.nextafter(s[allowed],np.inf))
