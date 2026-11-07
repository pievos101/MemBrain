"""Unsupervised full-curve clustering from text-only completion transcripts."""
import numpy as np
from membrain_text import words,similarity

GRID=np.linspace(.125,1.,8)
FLOOR=.001


def curve_features(short_record,long_record):
    if short_record['text']!=long_record['text']:raise ValueError('Candidate mismatch')
    n=len(words(short_record['text']));progress=[];whole=[];next_word=[]
    if len(short_record['queries'])!=len(long_record['queries']):raise ValueError('Query mismatch')
    for a,b in zip(short_record['queries'],long_record['queries']):
        if a['prefix']!=b['prefix']:raise ValueError('Prefix mismatch')
        progress.append(a['progress']/(1-4/n))
        whole.append(1-similarity(b['suffix'],b['answer']['greedy'])['word_edit'])
        target=words(a['suffix'])[0];samples=a['answer']['samples']
        hits=sum(bool(words(s)) and words(s)[0]==target for s in samples)
        next_word.append(1-hits/len(samples))
    return np.interp(GRID,progress,whole),np.interp(GRID,progress,next_word)


def early_cost(x):
    x=np.asarray(x)
    if x.shape[-1]==1:return x[...,0]
    if x.shape[-1]%len(GRID):raise ValueError('Invalid feature dimension')
    blocks=x.reshape(*x.shape[:-1],-1,len(GRID));w=1-GRID
    return (np.trapezoid(blocks*w,GRID,axis=-1)/np.trapezoid(w,GRID)).mean(-1)


def feature_sets(short_records,long_records):
    if len(short_records)!=len(long_records):raise ValueError('Candidate count mismatch')
    values=[curve_features(a,b) for a,b in zip(short_records,long_records)]
    whole=np.array([a for a,b in values]);next_word=np.array([b for a,b in values])
    return {'Whole curve':whole,'Next-word curve':next_word,'Joint curves':np.c_[whole,next_word],
      'Whole early scalar':early_cost(whole)[:,None],'Next-word early scalar':early_cost(next_word)[:,None],
      'Whole mean scalar':whole.mean(1,keepdims=True),'Next-word mean scalar':next_word.mean(1,keepdims=True)}


def _log_density(x,mu,var,weight):
    return np.log(weight)[None,:]-.5*np.sum(np.log(2*np.pi*var)[None,:,:]+(x[:,None,:]-mu[None,:,:])**2/var[None,:,:],axis=2)


def _responsibility(logp):
    maximum=logp.max(1);normal=maximum+np.log(np.exp(logp-maximum[:,None]).sum(1))
    return np.exp(logp-normal[:,None]),float(normal.sum())


def fit(x,max_components=2):
    """Fit raw bounded distance vectors. No labels, target statistics or rotations."""
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or len(x)<20 or not np.isfinite(x).all():raise ValueError('Need >=20 finite unlabeled vectors')
    n,d=x.shape;candidates=[];global_var=np.maximum(x.var(0),FLOOR)
    for k in range(1,max_components+1):
        best=None
        for seed in [3,17,29,53]:
            rng=np.random.default_rng(seed);chosen=[int(rng.integers(n))]
            while len(chosen)<k:
                distance=np.min(np.sum((x[:,None,:]-x[chosen][None,:,:])**2,axis=2),axis=1)
                chosen.append(int(distance.argmax()))
            mu=x[chosen].copy();var=np.tile(global_var,(k,1));weight=np.full(k,1/k);previous=-np.inf;valid=True
            for _ in range(200):
                resp,ll=_responsibility(_log_density(x,mu,var,weight));mass=resp.sum(0)
                if (mass<2).any():valid=False;break
                mu=resp.T@x/mass[:,None]
                var=np.maximum(resp.T@(x*x)/mass[:,None]-mu*mu,FLOOR)
                weight=np.maximum(mass/n,.05);weight/=weight.sum()
                if abs(ll-previous)<1e-7:break
                previous=ll
            if not valid:continue
            _,ll=_responsibility(_log_density(x,mu,var,weight))
            if best is None or ll>best[0]:best=(ll,mu.copy(),var.copy(),weight.copy())
        if best is None:continue
        ll,mu,var,weight=best;order=np.argsort(early_cost(mu),kind='stable')
        parameters=2*d*k+k-1;bic=parameters*np.log(n)-2*ll
        candidates.append(dict(components=k,means=mu[order].tolist(),variances=var[order].tolist(),weights=weight[order].tolist(),
             log_likelihood=ll,bic=float(bic),parameters=parameters))
    selected=min(candidates,key=lambda m:(m['bic'],m['components']))
    return dict(**selected,dimension=d,fit_count=n,variance_floor=FLOOR,
       candidates=[dict(components=m['components'],bic=m['bic']) for m in candidates],
       orientation='lowest early distance is highest recovery')


def predict(model,x):
    x=np.asarray(x,dtype=float)
    if model['components']==1:return np.full(len(x),.5),np.full(len(x),-1,dtype=int)
    resp,_=_responsibility(_log_density(x,np.array(model['means']),np.array(model['variances']),np.array(model['weights'])))
    high=resp[:,0]
    return high,(high>=.5).astype(int)
