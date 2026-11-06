"""Exploratory fixed-tail control: remove the changing-suffix pooling confound."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch.nn import functional as F
from spw_transformer import TinyTransformer
from spw_methods import evaluate_scores
from run_membrain_unsupervised import sha
from membrain_unsupervised import FRACTIONS,early_area


@torch.no_grad()
def extract(model,sequences,fractions=FRACTIONS,tail=4):
    model.eval();x=torch.tensor(np.asarray(sequences),dtype=torch.long);n,length=x.shape
    if length-max(round(length*f) for f in fractions)<tail:raise ValueError('Tail must remain after every prefix')
    keys=['true_embedding','generated_embedding','cosine','teacher_ce','rollout_ce','accuracy']
    result={k:[] for k in keys}
    for f in fractions:
        p=round(length*f);parts={k:[] for k in keys}
        for start in range(0,n,64):
            b=x[start:start+64];tl,th,_=model(b,return_hidden=True)
            tc=F.cross_entropy(tl[:,length-tail-1:length-1].transpose(1,2),b[:,-tail:],reduction='none').mean(1)
            logits,_,cache=model(b[:,:p],return_hidden=True);hidden=[];losses=[];tokens=[]
            for j in range(p,length):
                losses.append(F.cross_entropy(logits[:,-1],b[:,j],reduction='none'))
                token=logits[:,-1].argmax(-1);tokens.append(token)
                logits,h,cache=model(token[:,None],cache=cache,return_hidden=True);hidden.append(h[:,0])
            a=th[:,-tail:].mean(1);g=torch.stack(hidden,1)[:,-tail:].mean(1)
            values=dict(true_embedding=a,generated_embedding=g,cosine=F.cosine_similarity(a,g),teacher_ce=tc,
                rollout_ce=torch.stack(losses,1)[:,-tail:].mean(1),accuracy=(torch.stack(tokens,1)[:,-tail:]==b[:,-tail:]).float().mean(1))
            for k,v in values.items():parts[k].append(v.numpy())
        for k in keys:result[k].append(np.concatenate(parts[k]))
    return {k:np.stack(v,axis=1) for k,v in result.items()}


def score(curves,fractions=FRACTIONS):
    return {'Fixed-tail embedding recovery':early_area(curves['cosine'],fractions),
            'Fixed-tail rollout loss':-early_area(curves['rollout_ce'],fractions),
            'Fixed-tail teacher loss':-early_area(curves['teacher_ce'],fractions)}


def main():
    torch.set_num_threads(4);root=Path('results/membrain_shape');protocol=Path(__file__).with_name('MEMBRAIN_TAIL_CONTROL.md')
    frozen=root/'tail_control_freeze.json';hashes=dict(code=sha(Path(__file__)),protocol=sha(protocol))
    if frozen.exists():
        if json.loads(frozen.read_text())['hashes']!=hashes:raise ValueError('Control code changed')
    else:frozen.write_text(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,exploratory=True),indent=2)+'\n')
    with np.load(root/'benchmark_data.npz') as z:pools={k:z[k].copy() for k in ['ag_news','wikitext2']}
    paths=sorted((root/'cells').glob('*/metrics.json'))
    if len(paths)!=12:raise RuntimeError('Complete all12 primary targets first')
    for path in paths:
        output=path.parent/'tail_control_metrics.json'
        if output.exists():continue
        r=json.loads(path.read_text())
        with np.load(path.parent/'scores.npz') as z:ids=z['eval_ids'];y=z['labels']
        model=TinyTransformer();model.load_state_dict(torch.load(path.parent/'target.pt',weights_only=True));model.eval()
        curves=extract(model,pools[r['corpus']][ids]);scores=score(curves)
        np.savez_compressed(path.parent/'tail_unlabeled_scores.npz',**scores)
        displacement=float(np.max(np.abs(curves['true_embedding']-curves['true_embedding'][:,:1])))
        losschange=float(np.max(np.abs(curves['teacher_ce']-curves['teacher_ce'][:,:1])))
        if displacement>1e-6 or losschange>1e-6:raise AssertionError('Teacher-forced fixed-tail invariance failed')
        np.savez_compressed(path.parent/'tail_control.npz',labels=y,**curves,**scores)
        output.write_text(json.dumps(dict(exploratory=True,tail_tokens=4,maximum_true_embedding_change=displacement,
             maximum_teacher_loss_change=losschange,metrics={k:evaluate_scores(y,v) for k,v in scores.items()}),indent=2)+'\n')
        print('CONTROL',path.parent.name,'true-trajectory displacement',displacement,flush=True)


if __name__=='__main__':main()
