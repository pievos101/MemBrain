"""Two suffix-embedding trajectories, with absolute true-token loss. No attack fit."""
import numpy as np
import torch
from torch.nn import functional as F
from membrain_unsupervised import FRACTIONS,early_area


def score(curves,fractions=FRACTIONS):
    distance=np.clip((1-np.asarray(curves['cosine']))/2,0,1)
    teacher=np.asarray(curves['teacher_ce']);rollout=np.asarray(curves['rollout_ce'])
    # Fixed nats-per-token cost: confident true prediction + recovery mismatch.
    return {'Membrain':-early_area(teacher+distance*rollout,fractions),
      'Teacher loss curve':-early_area(teacher,fractions),
      'Rollout loss curve':-early_area(rollout,fractions),
      'Embedding curve':-early_area(distance,fractions),
      'Membrain without rollout weighting':-early_area(teacher+distance,fractions),
      'Loss curves without embeddings':-early_area((teacher+rollout)/2,fractions)}


@torch.no_grad()
def extract(model,sequences,fractions=FRACTIONS):
    model.eval();x=torch.tensor(np.asarray(sequences),dtype=torch.long);n,length=x.shape
    keys=['true_embedding','generated_embedding','cosine','teacher_ce','rollout_ce','accuracy','exact']
    result={k:[] for k in keys}
    for fraction in fractions:
        p=max(1,min(length-1,round(length*fraction)));parts={k:[] for k in keys}
        for start in range(0,n,64):
            b=x[start:start+64];tl,th,_=model(b,return_hidden=True)
            teacher=F.cross_entropy(tl[:,p-1:length-1].transpose(1,2),b[:,p:],reduction='none').mean(1)
            logits,_,cache=model(b[:,:p],return_hidden=True);hidden=[];losses=[];tokens=[]
            for j in range(p,length):
                losses.append(F.cross_entropy(logits[:,-1],b[:,j],reduction='none'))
                token=logits[:,-1].argmax(-1);tokens.append(token)
                logits,h,cache=model(token[:,None],cache=cache,return_hidden=True);hidden.append(h[:,0])
            true=th[:,p:].mean(1);generated=torch.stack(hidden,1).mean(1)
            match=torch.stack(tokens,1)==b[:,p:]
            values=dict(true_embedding=true,generated_embedding=generated,cosine=F.cosine_similarity(true,generated),
                teacher_ce=teacher,rollout_ce=torch.stack(losses,1).mean(1),accuracy=match.float().mean(1),exact=match.all(1).float())
            for k,v in values.items():parts[k].append(v.numpy())
        for k in keys:result[k].append(np.concatenate(parts[k]))
    return {k:np.stack(v,axis=1) for k,v in result.items()}
