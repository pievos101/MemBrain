"""Fit full-curve mixtures from cached strings; labels are loaded only afterward."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from membrain_curve_cluster import feature_sets,fit,predict,early_cost
from spw_methods import evaluate_scores

FAMILIES=['Whole curve','Next-word curve','Joint curves','Whole early scalar','Next-word early scalar','Whole mean scalar','Next-word mean scalar']
PRIMARY='Whole curve'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def group_metrics(y,pred):
    if (pred<0).all():return None
    positive=pred==1;tp=int(np.sum(positive&(y==1)));fp=int(np.sum(positive&(y==0)))
    npos=int(y.sum());nneg=len(y)-npos
    if not npos or not nneg:return None
    return dict(precision=tp/max(1,tp+fp),recall=tp/npos,false_positive_rate=fp/nneg,
                balanced_accuracy=.5*(tp/npos+1-fp/nneg),predicted_high=int(positive.sum()),count=len(y))


def intervals(y,scores,seed,count=1000):
    rng=np.random.default_rng(seed);pos=np.flatnonzero(y);neg=np.flatnonzero(1-y)
    draws=[np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))] for _ in range(count)]
    def auc(labels,values):
        order=np.argsort(values,kind='stable');s=values[order];lab=labels[order]
        starts=np.r_[0,np.flatnonzero(np.diff(s))+1];ends=np.r_[starts[1:],len(s)]
        ranks=np.repeat((starts+ends+1)/2,ends-starts);n=labels.sum()
        return (ranks[lab==1].sum()-n*(n+1)/2)/(n*(len(labels)-n))
    boot={m:np.array([auc(y[d],s[d]) for d in draws]) for m,s in scores.items()}
    point={m:evaluate_scores(y,s)['auc'] for m,s in scores.items()}
    return {m:dict(auc_ci95=np.quantile(v,[.025,.975]).tolist(),differences={ref:dict(delta=point[m]-point[ref],
      ci95=np.quantile(v-boot[ref],[.025,.975]).tolist()) for ref in ['Whole early scalar','Next-word early scalar']}) for m,v in boot.items()}


def run_cell(source,root):
    path=root/'cells'/source.name;path.mkdir(parents=True,exist_ok=True)
    if (path/'metrics.json').exists():return
    def read(name):
        with (source/name).open() as f:return [json.loads(line) for line in f]
    short=read('primary_transcripts.jsonl');long=read('greedy_transcripts.jsonl')
    features=feature_sets(short,long)
    old_split=json.loads((source/'unlabeled_mixtures.json').read_text())
    discovery=np.array(old_split['discovery']);heldout=np.array(old_split['heldout'])
    assert len(discovery)==len(heldout)==128 and not(set(discovery)&set(heldout))
    models={};probabilities={};assignments={}
    for name in FAMILIES:
        models[name]=fit(features[name][discovery]);probabilities[name],assignments[name]=predict(models[name],features[name][heldout])
    name='Whole curve up-to3';models[name]=fit(features[PRIMARY][discovery],max_components=3)
    probabilities[name],assignments[name]=predict(models[name],features[PRIMARY][heldout])
    probabilities['Whole raw recovery']=-early_cost(features['Whole curve'][heldout])
    probabilities['Next-word raw recovery']=-early_cost(features['Next-word curve'][heldout])
    # All feature matrices, mixtures and held-out outputs are fixed BEFORE labels.
    np.savez_compressed(path/'unlabeled_features.npz',**features)
    save(path/'unlabeled_models.json',dict(discovery=discovery.tolist(),heldout=heldout.tolist(),models=models))
    np.savez_compressed(path/'unlabeled_predictions.npz',**probabilities)
    with np.load(source/'scores.npz') as z:y=z['labels'][heldout];high=z['high_exposure'][heldout]
    owner=json.loads((source/'metrics.json').read_text());boot=intervals(y,probabilities,owner['seed'])
    metrics={m:dict(**evaluate_scores(y,p),**boot[m]) for m,p in probabilities.items()}
    groups={m:dict(components=models[m]['components'],membership=group_metrics(y,pred),
                  high_exposure=group_metrics(high,pred) if high.any() else None) for m,pred in assignments.items()}
    exposure={m:evaluate_scores(high,p) for m,p in probabilities.items()} if high.any() else None
    np.savez_compressed(path/'evaluation.npz',labels=y,high_exposure=high,**probabilities)
    save(path/'metrics.json',dict(capacity=owner['capacity'],corpus=owner['corpus'],seed=owner['seed'],regime=owner['regime'],
       metrics=metrics,groups=groups,high_exposure_metrics=exposure,heldout_members=int(y.sum()),heldout_nonmembers=int((1-y).sum()),
       exploratory_reused_data=True,membership_supervised=False,new_model_queries=0,
       transcript_sha256=dict(short=sha(source/'primary_transcripts.jsonl'),long=sha(source/'greedy_transcripts.jsonl'))))
    print(source.name,{m:round(metrics[m]['auc'],4) for m in [PRIMARY,'Whole early scalar','Next-word curve','Next-word early scalar','Joint curves']},
          'k',{m:models[m]['components'] for m in models},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('results/membrain_text_validation'))
    p.add_argument('--output',type=Path,default=Path('results/membrain_curve_cluster'));args=p.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=True);directory=Path(__file__).parent
    hashes={n:sha(directory/n) for n in ['membrain_curve_cluster.py','run_membrain_curve_cluster.py','MEMBRAIN_CURVE_CLUSTER_PROTOCOL.md']}
    frozen=root/'frozen_protocol.json'
    if frozen.exists():
        if json.loads(frozen.read_text())['hashes']!=hashes:raise ValueError('Use a new output after code changes')
    else:save(frozen,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,exploratory=True,new_model_queries=0))
    paths=sorted((args.source/'cells').glob('*/metrics.json'))
    if len(paths)!=16:raise ValueError('Need16 cached validation targets')
    for path in paths:run_cell(path.parent,root)


if __name__=='__main__':main()
