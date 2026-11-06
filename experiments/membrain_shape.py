"""Label-free geometry of two suffix-embedding trajectories. No fitted attack."""
import numpy as np
import torch
from torch.nn import functional as F
from membrain_unsupervised import FRACTIONS,early_area


def cos(a,b):
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    # No movement / collapsed trajectories provide no directional evidence.
    return np.divide(np.sum(a*b,axis=-1),den,out=np.zeros_like(den,dtype=float),where=den>1e-10).clip(-1,1)


def geometry(a,b,fractions=FRACTIONS):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-10)
    b=b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-10)
    ca=a-a.mean(1,keepdims=True);cb=b-b.mean(1,keepdims=True)
    centered=cos(ca.reshape(len(a),-1),cb.reshape(len(b),-1))
    da=np.diff(a,axis=1);db=np.diff(b,axis=1)
    mid=(np.asarray(fractions[:-1])+np.asarray(fractions[1:]))/2
    velocity=early_area(cos(da,db),mid)
    proximity=early_area(cos(a,b),fractions)
    la=np.linalg.norm(da,axis=-1).sum(1);lb=np.linalg.norm(db,axis=-1).sum(1)
    length_agreement=-np.abs(np.log((la+1e-6)/(lb+1e-6)))
    return dict(shape=(centered+velocity+proximity)/3,centered=centered,velocity=velocity,
                proximity=proximity,path_length_agreement=length_agreement)


def score(curves,token_logp,fractions=FRACTIONS):
    uniform=geometry(curves['true_embedding'],curves['generated_embedding'],fractions)
    hard=geometry(curves['hard_true_embedding'],curves['hard_generated_embedding'],fractions)
    lp=np.asarray(token_logp);k=max(1,int(np.ceil(.2*lp.shape[1])))
    mink=np.sort(lp,axis=1)[:,:k].mean(1)
    result={'Membrain shape hybrid':mink+.5*hard['shape'],
      'Membrain hard-token geometry':hard['shape'],'Membrain uniform geometry':uniform['shape'],
      'Hard-token teacher loss curve':-early_area(curves['hard_teacher_ce'],fractions),
      'Hard-token rollout loss curve':-early_area(curves['hard_rollout_ce'],fractions),
      'Mean teacher loss curve':-early_area(curves['teacher_ce'],fractions),
      'Mean rollout loss curve':-early_area(curves['rollout_ce'],fractions),
      'Min-k20%':mink,'LOSS':lp.mean(1)}
    for prefix,g in [('Uniform',uniform),('Hard-token',hard)]:
        for name in ['centered','velocity','proximity','path_length_agreement']:
            result[f'{prefix} {name}']=g[name]
    return result


@torch.no_grad()
def extract(model,sequences,fractions=FRACTIONS):
    model.eval();x=torch.tensor(np.asarray(sequences),dtype=torch.long);n,length=x.shape
    keys=['true_embedding','generated_embedding','hard_true_embedding','hard_generated_embedding',
          'teacher_ce','rollout_ce','hard_teacher_ce','hard_rollout_ce','accuracy','exact']
    result={k:[] for k in keys}
    for f in fractions:
        p=max(1,min(length-1,round(length*f)));parts={k:[] for k in keys};count=max(1,int(np.ceil(.2*(length-p))))
        for start in range(0,n,64):
            b=x[start:start+64];tl,th,_=model(b,return_hidden=True)
            tc=F.cross_entropy(tl[:,p-1:length-1].transpose(1,2),b[:,p:],reduction='none')
            # Same teacher-defined hard positions pool BOTH origins; no generated self-loss selection.
            chosen=tc.topk(count,dim=1,sorted=False).indices
            logits,_,cache=model(b[:,:p],return_hidden=True);hidden=[];losses=[];tokens=[]
            for j in range(p,length):
                losses.append(F.cross_entropy(logits[:,-1],b[:,j],reduction='none'))
                token=logits[:,-1].argmax(-1);tokens.append(token)
                logits,h,cache=model(token[:,None],cache=cache,return_hidden=True);hidden.append(h[:,0])
            true=th[:,p:];generated=torch.stack(hidden,1);rc=torch.stack(losses,1)
            gather=chosen[:,:,None].expand(-1,-1,true.shape[-1]);match=torch.stack(tokens,1)==b[:,p:]
            values=dict(true_embedding=true.mean(1),generated_embedding=generated.mean(1),
                hard_true_embedding=true.gather(1,gather).mean(1),hard_generated_embedding=generated.gather(1,gather).mean(1),
                teacher_ce=tc.mean(1),rollout_ce=rc.mean(1),hard_teacher_ce=tc.gather(1,chosen).mean(1),
                hard_rollout_ce=rc.gather(1,chosen).mean(1),accuracy=match.float().mean(1),exact=match.all(1).float())
            for k,v in values.items():parts[k].append(v.numpy())
        for k in keys:result[k].append(np.concatenate(parts[k]))
    return {k:np.stack(v,axis=1) for k,v in result.items()}
