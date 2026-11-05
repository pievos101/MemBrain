"""Membrain classifier: class 0 true continuations, class 1 generated suffixes.

Train/evaluate with no membership supervision, then inspect exposure association.
CLI inference: --text-file FILE --model-seed 51001 --variant trajectory.
"""
import argparse
import copy
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import membrain_recovery as recovery
from membrain_loss_curves import auc

ROOT=Path(__file__).resolve().parents[1]
RECOVERY=ROOT/'results/membrain_recovery'
VARIANTS=['trajectory','pooled','single90']
SCORING='whole_suffix_hidden_trajectory_origin_classes_v1'


def auxiliary_sentences():
    """Independent texts excluded from ALL recovery evaluation/background texts."""
    data=json.loads((RECOVERY/'sentences.json').read_text())
    excluded=set(data['evaluation']+data['background'])
    text=re.sub(r'\s+',' ',recovery.previous.SOURCE.read_text())
    candidates=[];seen=set(excluded)
    for part in re.split(r'(?<=[.!?])\s+',text):
        part=re.sub(r'^[A-Za-z][A-Za-z ]{0,35}:\s*','',part).strip()
        part=''.join(c if c in recovery.ALPHABET else ' ' for c in part)
        part=re.sub(r'\s+',' ',part)
        if 80<=len(part)<=120 and part[-1:] in '.!?' and part not in seen:
            seen.add(part);candidates.append(part)
    rng=np.random.default_rng(72013)
    chosen=[candidates[i] for i in rng.choice(len(candidates),320,replace=False)]
    assert not set(chosen)&excluded
    assert len(set(chosen))==320
    return chosen


class OriginClassifier(nn.Module):
    def __init__(self,variant,k=12):
        super().__init__();self.variant=variant
        if variant=='trajectory':
            self.conv=nn.Sequential(nn.Conv1d(96,32,3,padding=1),nn.ReLU(),
                                    nn.Conv1d(32,16,3,padding=1),nn.ReLU())
            self.head=nn.Sequential(nn.Flatten(),nn.Linear(16*k,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,1))
        else:self.head=nn.Sequential(nn.Linear(96,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,1))

    def forward(self,x):
        if self.variant=='trajectory':return self.head(self.conv(x.transpose(1,2)))[:,0]
        return self.head(x.mean(1) if self.variant=='pooled' else x[:,-1])[:,0]


def origin_loss(model,o,g):
    """Balanced class prior despite multiple generated paths per document."""
    lo=model(o);lg=model(g.reshape(-1,g.shape[-2],g.shape[-1]))
    return .5*(F.binary_cross_entropy_with_logits(lo,torch.zeros_like(lo))+
               F.binary_cross_entropy_with_logits(lg,torch.ones_like(lg)))


def fit_classifier(original,generated,variant,seed):
    torch.manual_seed(seed)
    # Only auxiliary training sentences fit normalization and classifier weights.
    train_o=original[:160];train_g=generated[:160,1:5]
    mean=.5*(train_o.mean((0,1))+train_g.mean((0,1,2)))
    second=.5*((train_o**2).mean((0,1))+(train_g**2).mean((0,1,2)))
    sd=np.sqrt(np.maximum(second-mean**2,1e-4))
    o=torch.tensor((original-mean)/sd);g=torch.tensor((generated[:,1:]-mean)/sd)
    model=OriginClassifier(variant);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.01)
    best,saved,stale=float('inf'),None,0;history=[]
    for _ in range(160):
        model.train()
        for ids in torch.randperm(160).split(32):
            loss=origin_loss(model,o[ids],g[ids,:4]);opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():value=origin_loss(model,o[160:200],g[160:200,:4]).item()
        history.append(value)
        if value<best-1e-5:best,saved,stale=value,copy.deepcopy(model.state_dict()),0
        else:stale+=1
        if stale>=15:break
    model.load_state_dict(saved);model.eval()
    # Separate auxiliary calibration set, not the early-stopping set.
    with torch.no_grad():
        co=model(o[200:240]).double()
        cg=model(g[200:240].reshape(-1,12,96)).double()
    scale_raw=torch.zeros((),dtype=torch.float64,requires_grad=True)
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    calopt=torch.optim.LBFGS([scale_raw,bias],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        calopt.zero_grad();a=F.softplus(scale_raw)
        loss=.5*(F.binary_cross_entropy_with_logits(a*co+bias,torch.zeros_like(co))+
                  F.binary_cross_entropy_with_logits(a*cg+bias,torch.ones_like(cg)))+.001*(a-1)**2
        loss.backward();return loss
    calopt.step(closure)
    return dict(variant=variant,state=saved,mean=torch.tensor(mean),sd=torch.tensor(sd),
                scale=F.softplus(scale_raw).detach(),bias=bias.detach(),
                class_labels={'0':'true prefix + true continuation','1':'true prefix + generated suffix'},
                history=history,scoring=SCORING)


@torch.no_grad()
def predict(checkpoint,original,generated):
    model=OriginClassifier(checkpoint['variant']);model.load_state_dict(checkpoint['state']);model.eval()
    o=(torch.as_tensor(original)-checkpoint['mean'])/checkpoint['sd']
    g=(torch.as_tensor(generated)-checkpoint['mean'])/checkpoint['sd']
    lo=model(o);lg=model(g.reshape(-1,12,96)).reshape(g.shape[:2])
    po=(checkpoint['scale']*lo+checkpoint['bias']).sigmoid().numpy()
    pg=(checkpoint['scale']*lg+checkpoint['bias']).sigmoid().numpy()
    return po,pg  # Positive class is generated (class 1).


def load_lm(seed):
    lm=recovery.SentenceLM()
    lm.load_state_dict(torch.load(RECOVERY/f'model_{seed}.pt',weights_only=True));lm.eval()
    return lm


def inference(args):
    text=re.sub(r'\s+',' ',args.text_file.read_text()).strip()
    if not 80<=len(text)<=120 or any(c not in recovery.ALPHABET for c in text):
        raise ValueError('This experimental classifier expects one 80–120 character printable-ASCII sentence.')
    path=args.output/f'classifier_{args.model_seed}_{args.variant}.pt'
    checkpoint=torch.load(path,weights_only=True)
    lm=load_lm(args.model_seed);torch.manual_seed(args.model_seed+30000)
    r,_=recovery.recover(lm,[text],args.samples,return_embeddings=True)
    po,pg=predict(checkpoint,r['original_trajectory'],r['generated_trajectories'][:,1:])
    result=dict(text=text,model_seed=args.model_seed,variant=args.variant,
                p_class_0_true_continuation=1.0-float(po[0]),p_class_1_generated_suffix=float(po[0]),
                mean_p_class_1_generated_reference=float(pg.mean()),
                paired_class1_gap=float(po[0]-pg.mean()),
                interpretation='Origin-class probabilities under balanced auxiliary class prior; not calibrated training-membership probabilities.')
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/membrain_trajectory_classifier')
    p.add_argument('--samples',type=int,default=8);p.add_argument('--text-file',type=Path)
    p.add_argument('--model-seed',type=int,default=51001);p.add_argument('--variant',choices=VARIANTS,default='trajectory')
    args=p.parse_args();torch.set_num_threads(2)
    if args.text_file:inference(args);return
    assert args.samples>=4
    args.output.mkdir(parents=True,exist_ok=True)
    aux=auxiliary_sentences();evaluation=json.loads((RECOVERY/'sentences.json').read_text())['evaluation']
    texts=aux+evaluation
    (args.output/'auxiliary_sentences.json').write_text(json.dumps(dict(train=aux[:160],early_stop=aux[160:200],
            calibration=aux[200:240],holdout=aux[240:320]),indent=2))
    (args.output/'example_new_sentence.txt').write_text(aux[240])
    auxiliary_rows=[];exposure_rows=[];membership_rows=[]
    for seed in range(51001,51005):
        path=args.output/f'features_{seed}.npz'
        if path.exists():
            with np.load(path) as f:r={k:f[k] for k in f.files}
            assert int(r['samples'])==args.samples
        else:
            print(f'Extracting full-suffix embedding trajectories: LM {seed}',flush=True)
            lm=load_lm(seed);torch.manual_seed(seed+20000)
            raw,_=recovery.recover(lm,texts,args.samples,return_embeddings=True)
            r=dict(original=raw['original_trajectory'],generated=raw['generated_trajectories'],
                   loss=raw['teacher_full_loss'],samples=np.array(args.samples))
            np.savez_compressed(path,**r)
        original,generated=r['original'],r['generated']
        with np.load(RECOVERY/f'model_{seed}.npz') as f:repeats=f['repeats']
        predictions={}
        for variant in VARIANTS:
            print(f'Training origin classes: LM {seed}, {variant}',flush=True)
            checkpoint=fit_classifier(original,generated,variant,seed+len(variant))
            torch.save(checkpoint,args.output/f'classifier_{seed}_{variant}.pt')
            po,pg=predict(checkpoint,original,generated[:,1:])
            greedy_o,greedy_g=predict(checkpoint,original,generated[:,:1])
            # Final auxiliary holdout; class balance weights documents equally.
            h=slice(240,320);a=po[h];b=pg[h]
            auxiliary_rows.append(dict(seed=seed,variant=variant,
                balanced_accuracy=.5*((a<.5).mean()+(b>=.5).mean()),
                origin_auc=auc(b.ravel(),a),balanced_brier=.5*(np.mean(a*a)+np.mean((b-1)**2))))
            eo,eg=po[320:],pg[320:];gap=eo-eg.mean(1)
            predictions[f'{variant}_p_generated_original']=eo
            predictions[f'{variant}_p_generated_reference']=eg
            predictions[f'{variant}_paired_gap']=gap
            predictions[f'{variant}_greedy_gap']=eo-greedy_g[320:,0]
            for level in recovery.LEVELS:
                mask=repeats==level
                exposure_rows.append(dict(seed=seed,variant=variant,presentations=int(level*40),
                    mean_p_generated_original=eo[mask].mean(),mean_p_generated_reference=eg[mask].mean(),
                    mean_paired_gap=gap[mask].mean(),mean_greedy_gap=(eo-greedy_g[320:,0])[mask].mean()))
            # Predefined positive direction: originals looking generated-like.
            # Membership labels are used ONLY for these evaluation summaries.
            for level in recovery.LEVELS[1:]:
                member=repeats==level;non=repeats==0
                for score_name,score in [('class1_probability',eo),('paired_gap',gap),
                    ('greedy_gap',eo-greedy_g[320:,0]),('LOSS_full_sentence',-r['loss'][320:])]:
                    membership_rows.append(dict(seed=seed,variant=variant,presentations=int(level*40),
                        score=score_name,auc=auc(score[member],score[non])))
        np.savez_compressed(args.output/f'predictions_{seed}.npz',**predictions)
        print(f'Completed LM {seed}',flush=True)
    for name,rows in [('auxiliary_holdout',auxiliary_rows),('exposure_scores',exposure_rows),('membership_auc',membership_rows)]:
        with (args.output/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output/'config.json').write_text(json.dumps(dict(scoring=SCORING,samples=args.samples,
        class_0='true prefix + true continuation',class_1='true prefix + generated suffix',
        feature_shape=[12,96],fractions=recovery.FRACTIONS.tolist(),variants=VARIANTS,
        auxiliary_split_sizes=[160,40,40,80],membership_labels_used_for_fitting=False,
        generation='entire remaining sentence, sampled temperature 1; greedy reference diagnostic',
        seeds=list(range(51001,51005)),torch=torch.__version__,numpy=np.__version__),indent=2))
    print('DONE',flush=True)


if __name__=='__main__':main()
