"""Membrain: a fixed, per-text membership score, with no fitted attack model.

Only target-model outputs and the candidate text enter this module. In
particular there is no origin classifier, membership-label argument, learned
normalizer, shadow model, or membership-probability calibration.
"""
import numpy as np
import torch
from torch.nn import functional as F

FRACTIONS=(.05,.1,.2,.3,.4,.5,.6,.7,.8,.9)
EPS=1e-8


def position_weights(gain):
    g=np.clip(np.asarray(gain,dtype=np.float64),-20,20)
    p=np.exp(g-g.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    return .5/g.shape[-1]+.5*p


def corrected_alignment(aligned,null):
    """Within-text null removes generic similarity; collapsed states give no cue."""
    a=np.asarray(aligned);b=np.asarray(null)
    return np.clip((a-b)/np.maximum(1-b,1e-6),0,1)


def early_area(curve,fractions=FRACTIONS):
    f=np.asarray(fractions,dtype=np.float64)
    c=np.asarray(curve,dtype=np.float64)
    if c.shape[-1]!=len(f) or len(f)<2 or not np.all(np.diff(f)>0):raise ValueError('Invalid prefix grid')
    density=1-f
    return np.trapezoid(c*density,f,axis=-1)/np.trapezoid(density,f)


def recovery_support(aligned,null,rollout_ce,teacher_ce):
    alignment=corrected_alignment(aligned,null)
    excess=np.maximum(np.asarray(rollout_ce)-np.asarray(teacher_ce),0)/(1+np.asarray(teacher_ce))
    # Equal fixed embedding and probability attenuation; no fitted coefficients.
    return (.5+.5*alignment)/(1+excess),alignment,excess


def score_curves(curves,token_logp,fractions=FRACTIONS):
    """Higher score => stronger membership evidence; this is NOT a probability."""
    lp=np.asarray(token_logp,dtype=np.float64)
    k=max(1,int(np.ceil(.2*lp.shape[1])))
    hard=np.sort(lp,axis=1)[:,:k].mean(1)
    support,alignment,excess=recovery_support(curves['aligned'],curves['null'],curves['rollout_ce'],curves['teacher_ce'])
    uniform,_,_=recovery_support(curves['uniform_aligned'],curves['uniform_null'],
                                curves['uniform_rollout_ce'],curves['uniform_teacher_ce'])
    recovery=np.log(np.maximum(early_area(support,fractions),EPS))
    no_embedding=np.log(np.maximum(early_area(1/(1+excess),fractions),EPS))
    embedding_only=np.log(np.maximum(early_area(.5+.5*alignment,fractions),EPS))
    return {'Membrain':hard+recovery,'Min-k20%':hard,'LOSS':lp.mean(1),
      'Membrain recovery only':recovery,'Membrain without embeddings':hard+no_embedding,
      'Membrain without rollout loss':hard+embedding_only,
      'Membrain uniform weights':hard+np.log(np.maximum(early_area(uniform,fractions),EPS))}


@torch.no_grad()
def extract_curves(model,sequences,spw_details,fractions=FRACTIONS):
    """Greedy whole-suffix rollouts; CE targets true tokens, never generated ones."""
    model.eval();x=torch.tensor(np.asarray(sequences),dtype=torch.long);n,length=x.shape
    if min(round(length*f) for f in fractions)<2:raise ValueError('Need prefixes of at least two tokens')
    gain=np.array([np.array(d['target_logp'])-np.array(d['shortened_logp']) for d in spw_details])
    keys=['aligned','null','teacher_ce','rollout_ce','uniform_aligned','uniform_null',
          'uniform_teacher_ce','uniform_rollout_ce','accuracy','exact','pooled_cosine']
    result={k:[] for k in keys}
    for fraction in fractions:
        p=round(length*fraction);parts={k:[] for k in keys}
        for start in range(0,n,64):
            b=x[start:start+64];tl,th,_=model(b,return_hidden=True)
            tc=F.cross_entropy(tl[:,p-1:length-1].transpose(1,2),b[:,p:],reduction='none')
            logits,_,cache=model(b[:,:p],return_hidden=True);hidden=[];ce=[];tokens=[]
            for j in range(p,length):
                ce.append(F.cross_entropy(logits[:,-1],b[:,j],reduction='none'))
                token=logits[:,-1].argmax(-1);tokens.append(token)
                logits,h,cache=model(token[:,None],cache=cache,return_hidden=True);hidden.append(h[:,0])
            gh=torch.stack(hidden,1);truth=th[:,p:];rc=torch.stack(ce,1)
            aligned=F.cosine_similarity(gh,truth,dim=-1)
            null=F.cosine_similarity(gh,torch.roll(truth,shifts=max(1,(length-p)//2),dims=1),dim=-1)
            w=torch.tensor(position_weights(gain[start:start+len(b),p-2:]),dtype=rc.dtype)
            match=torch.stack(tokens,1)==b[:,p:]
            values=dict(aligned=(w*aligned).sum(1),null=(w*null).sum(1),teacher_ce=(w*tc).sum(1),rollout_ce=(w*rc).sum(1),
                uniform_aligned=aligned.mean(1),uniform_null=null.mean(1),uniform_teacher_ce=tc.mean(1),uniform_rollout_ce=rc.mean(1),
                accuracy=match.float().mean(1),exact=match.all(1).float(),pooled_cosine=F.cosine_similarity(gh.mean(1),truth.mean(1)))
            for k,v in values.items():parts[k].append(v.numpy())
        for k in keys:result[k].append(np.concatenate(parts[k]))
    return {k:np.stack(v,axis=1) for k,v in result.items()}
