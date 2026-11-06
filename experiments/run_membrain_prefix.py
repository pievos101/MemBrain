"""Fit updated Membrain on shadows, freeze it, then run a fresh target benchmark."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import re
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from run_spw_benchmark import (SOURCES,FRACTIONS,make_neighbors,origin_classifier,save_json)
from spw_transformer import TinyTransformer,Oracle,train_lm
from spw_methods import spw_batch,evaluate_scores,zlib_score,lowercase_score,neighbor_score
from membrain_prefix import (recovery_features,assemble_features,feature_mask,fit_logistic,
                             decision,calibrate,probabilities,fpr_threshold)

VARIANTS=['Membrain','Probability-only learner','Membrain without embeddings','Membrain curves only']
FIT_SEEDS=[3101,3102,3103];VALID_SEED=3201;CAL_SEED=3301;TARGET_SEEDS=[4101,4102,4103]


def digest(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def prepare(source,root):
    if (root/'fresh_data.npz').exists():
        with np.load(root/'fresh_data.npz') as z:return {k:z[k].copy() for k in z.files}
    tok=Tokenizer.from_file(str(source/'tokenizer.json'));seen=set();pools={}
    with np.load(source/'benchmark_data.npz') as old:
        for key in old.files:seen.update(tuple(x) for x in old[key])
        bg=old['background'].copy()
    for name,spec in SOURCES.items():
        path=source/'data'/spec['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Source checksum')
        if name=='ag_news':
            with path.open() as f:texts=[r[2] for r in csv.reader(f)]
        else:texts=[s for s in path.read_text().splitlines() if not s.strip().startswith('=')]
        texts=list(dict.fromkeys(re.sub(r'\s+',' ',s).strip() for s in texts));texts=[s for s in texts if len(s)>=300]
        order=np.random.default_rng(8341).permutation(len(texts));texts=[texts[i] for i in order]
        for s in texts[:2048]:seen.add(tuple(tok.encode(s).ids[:48]))
        pools[name]=texts[2048:]
    arrays={'background':bg};manifest={}
    for name,texts in pools.items():
        chunks=[]
        for s in texts:
            ids=tok.encode(s).ids[:48];key=tuple(ids)
            if len(ids)!=48 or key in seen:continue
            seen.add(key);chunks.append(ids)
            if len(chunks)==2496:break
        if len(chunks)!=2496:raise ValueError(f'Need 2496 fresh chunks for {name}, found {len(chunks)}')
        x=np.asarray(chunks,dtype=np.int64);arrays[name]=x
        # Three fit pools, validation, calibration, target (384 each), then 192 origin auxiliaries.
        manifest[name]=dict(sha256=digest(x),roles={role:dict(start=start,end=end,sha256=digest(x[start:end]))
             for role,start,end in [('fit0',0,384),('fit1',384,768),('fit2',768,1152),
                 ('validation',1152,1536),('calibration',1536,1920),('target',1920,2304),('origin_aux',2304,2496)]})
    np.savez_compressed(root/'fresh_data.npz',**arrays);save_json(root/'fresh_manifest.json',manifest)
    (root/'tokenizer.json').write_bytes((source/'tokenizer.json').read_bytes())
    return arrays


def cell(root,source,tok,base,pool,aux,corpus,seed,regime,role):
    path=root/'cells'/f'{role}_{corpus}_{seed}_{regime}';path.mkdir(parents=True,exist_ok=True)
    if (path/'cell.json').exists():return path
    started=time.perf_counter();started_utc=datetime.now(timezone.utc).isoformat()
    order=np.random.default_rng(seed).permutation(384)
    members=order[:256];nonmembers=order[256:];eval_ids=np.r_[members[:128],nonmembers]
    y=np.r_[np.ones(128,dtype=int),np.zeros(128,dtype=int)];test=pool[eval_ids]
    model=copy.deepcopy(base);train=pool[members];epochs=4 if regime=='standard' else 12
    if regime=='repeated':train=np.concatenate([train,np.repeat(train[:32],7,axis=0)])
    print('CELL',path.name,flush=True);history=train_lm(model,train,epochs,seed+71)
    torch.save(model.state_dict(),path/'target.pt')
    words=[tok.decode([i],skip_special_tokens=False) for i in range(tok.get_vocab_size())]
    oracle=Oracle(model,words);lp=oracle.teacher(test)
    token_lp=np.take_along_axis(lp,test[:,1:,None],axis=2)[:,:,0];nll=-token_lp.mean(1)
    details=spw_batch(oracle,test,teacher_log_probs=lp,return_token_details=True);del lp
    curves,true,generated=recovery_features(model,test,details,FRACTIONS)
    features,names,groups=assemble_features(curves,details,token_lp,FRACTIONS)
    schema=dict(names=names,groups=groups)
    if (root/'feature_schema.json').exists():
        if json.loads((root/'feature_schema.json').read_text())!=schema:raise ValueError('Feature schema changed')
    else:save_json(root/'feature_schema.json',schema)
    scores={'LOSS':token_lp.mean(1),'Min-k20%':np.sort(token_lp,axis=1)[:,:10].mean(1),
            'SPW-MIA':np.array([d['spw'] for d in details])}
    diagnostic={}
    # Original Membrain and the more expensive remaining baselines only needed on fresh targets.
    if role=='target':
        from spw_transformer import trajectories
        at=trajectories(model,aux,FRACTIONS)
        predict,diagnostic,state=origin_classifier(at,seed+923)
        qt,qg=predict(true),predict(generated);scores['Membrain original']=-np.abs(qt-qg)
        torch.save(state,path/'origin_classifier.pt')
        texts=[tok.decode(row.tolist()) for row in test]
        lower_nll=oracle.nll([tok.encode(s.lower()).ids for s in texts])
        proposal=TinyTransformer();proposal.load_state_dict(torch.load(source/'proposal.pt',weights_only=True));proposal.eval()
        neighbor_tokens=make_neighbors(proposal,test,2871)
        neighbor_nll=oracle.nll(neighbor_tokens.reshape(-1,48)).reshape(len(test),10)
        scores.update(Zlib=np.array([zlib_score(a,b) for a,b in zip(nll,texts)]),
            Lowercase=np.array([lowercase_score(a,b) for a,b in zip(nll,lower_nll)]),
            Neighbor=np.array([neighbor_score(a,b) for a,b in zip(nll,neighbor_nll)]))
        np.savez_compressed(path/'trajectories.npz',labels=y,true=true,generated=generated,qtrue=qt,qgenerated=qg,**curves)
    np.savez_compressed(path/'features.npz',features=features,labels=y,eval_ids=eval_ids,**scores)
    save_json(path/'spw_details.json',details)
    save_json(path/'cell.json',dict(corpus=corpus,seed=seed,regime=regime,role=role,
       started_utc=started_utc,history=history,pool_sha256=digest(pool),membership_permutation_sha256=digest(order),
       origin_holdout=diagnostic,seconds=time.perf_counter()-started,oracle_stats=oracle.stats))
    return path


def load_cells(paths):
    data=[]
    for path in paths:
        with np.load(path/'features.npz') as z:data.append((z['features'].copy(),z['labels'].copy()))
    return np.concatenate([d[0] for d in data]),np.concatenate([d[1] for d in data])


def freeze(root,fit_paths,valid_paths,cal_paths):
    schema=json.loads((root/'feature_schema.json').read_text());models={};selection={}
    x,y=load_cells(fit_paths);vx,vy=load_cells(valid_paths);cx,cy=load_cells(cal_paths)
    for variant in VARIANTS:
        mask=feature_mask(schema['groups'],variant);candidates=[]
        for regularization in [.001,.01,.1,1.]:
            model=fit_logistic(x[:,mask],y,regularization);scores=decision(model,vx[:,mask])
            # Equal-cell macro AUC, not cross-target raw-score pooling.
            aucs=[evaluate_scores(vy[i:i+256],scores[i:i+256])['auc'] for i in range(0,len(vy),256)]
            candidates.append((float(np.mean(aucs)),regularization))
        # Prefer stronger regularization on exact ties.
        chosen=max(candidates)[1]
        model=fit_logistic(np.r_[x[:,mask],vx[:,mask]],np.r_[y,vy],chosen)
        cl=decision(model,cx[:,mask]);calibration=calibrate(cl,cy);pr=probabilities(calibration,cl)
        model.update(mask=mask,calibration=calibration,threshold=fpr_threshold(pr[cy==0]))
        models[variant]=model;selection[variant]=dict(candidates=candidates,chosen_regularization=chosen,
             calibration=calibration,calibration_threshold=model['threshold'],features=int(mask.sum()),
             calibration_metrics=evaluate_scores(cy,pr),calibration_prior=float(cy.mean()))
        print('FROZEN',variant,selection[variant],flush=True)
    # Persist the decision before ANY target model has been trained or evaluated.
    torch.save(models,root/'frozen_membrain.pt');save_json(root/'selection.json',selection)
    save_json(root/'freeze_manifest.json',dict(fit=[p.name for p in fit_paths],validation=[p.name for p in valid_paths],
         frozen_utc=datetime.now(timezone.utc).isoformat(),calibration=[p.name for p in cal_paths],selection_sha256=hashlib.sha256((root/'selection.json').read_bytes()).hexdigest(),
         model_sha256=hashlib.sha256((root/'frozen_membrain.pt').read_bytes()).hexdigest()))
    return models


def paired_intervals(y,scores,seed,count=1000):
    rng=np.random.default_rng(seed);pos=np.flatnonzero(y);neg=np.flatnonzero(1-y)
    draws=[np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))] for _ in range(count)]
    def auc(labels,values):
        order=np.argsort(values,kind='stable');s=values[order];lab=labels[order]
        starts=np.r_[0,np.flatnonzero(np.diff(s))+1];ends=np.r_[starts[1:],len(s)]
        ranks=np.repeat((starts+ends+1)/2,ends-starts);n=labels.sum()
        return (ranks[lab==1].sum()-n*(n+1)/2)/(n*(len(labels)-n))
    values={m:np.array([auc(y[d],s[d]) for d in draws]) for m,s in scores.items()}
    references=['LOSS','Min-k20%','SPW-MIA','Membrain original','Probability-only learner']
    return {m:dict(auc_ci95=np.quantile(v,[.025,.975]).tolist(),differences={
         ref:dict(delta=float(evaluate_scores(y,scores[m])['auc']-evaluate_scores(y,scores[ref])['auc']),
            ci95=np.quantile(v-values[ref],[.025,.975]).tolist()) for ref in references}) for m,v in values.items()}


def evaluate(path,models):
    with np.load(path/'features.npz') as z:
        y=z['labels'];x=z['features'];scores={k:z[k].copy() for k in z.files if k not in ['features','labels','eval_ids']}
    calibration_metrics={}
    for variant,m in models.items():
        scores[variant]=probabilities(m['calibration'],decision(m,x[:,m['mask']]))
        p=np.clip(scores[variant],1e-10,1-1e-10);pred=p>=m['threshold']
        calibration_metrics[variant]=dict(brier=float(np.mean((p-y)**2)),
          log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
          threshold=m['threshold'],fixed_threshold_tpr=float(pred[y==1].mean()),fixed_threshold_fpr=float(pred[y==0].mean()))
    metadata=json.loads((path/'cell.json').read_text());boot=paired_intervals(y,scores,metadata['seed'])
    metrics={k:dict(**evaluate_scores(y,v),**boot[k]) for k,v in scores.items()}
    rng=np.random.default_rng(712);imbalance={}
    for gamma in [.05,.1,.2,.4,.8,1.]:
        draws=[np.r_[rng.choice(np.flatnonzero(y),max(1,round(gamma*128)),replace=False),np.flatnonzero(1-y)] for _ in range(20)]
        imbalance[str(gamma)]={k:float(np.mean([evaluate_scores(y[d],v[d])['average_precision'] for d in draws])) for k,v in scores.items()}
    np.savez_compressed(path/'scores.npz',labels=y,**scores)
    save_json(path/'metrics.json',dict(**metadata,metrics=metrics,calibration=calibration_metrics,imbalance=imbalance))
    print('RESULT',path.name,{k:round(v['auc'],4) for k,v in metrics.items()},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('results/spw_scale'))
    p.add_argument('--output',type=Path,default=Path('results/membrain_prefix'));p.add_argument('--threads',type=int,default=4)
    p.add_argument('--stage',choices=['all','development','test'],default='all');args=p.parse_args()
    root=args.output;root.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    data=prepare(args.source,root);tok=Tokenizer.from_file(str(root/'tokenizer.json'))
    config=dict(name='Membrain',fit_seeds=FIT_SEEDS,validation_seed=VALID_SEED,calibration_seed=CAL_SEED,
       target_seeds=TARGET_SEEDS,fractions=FRACTIONS,regularizations=[.001,.01,.1,1.],variants=VARIANTS,
       fresh_documents_per_corpus=2496,architecture='same 182016-parameter Transformer as prior suite',
       python=platform.python_version(),torch=torch.__version__,numpy=np.__version__)
    save_json(root/'config.json',config)
    def base_for(seed):
        path=root/f'base_{seed}.pt';torch.manual_seed(seed);base=TinyTransformer()
        if path.exists():base.load_state_dict(torch.load(path,weights_only=True))
        else:train_lm(base,data['background'],3,seed);torch.save(base.state_dict(),path)
        base.eval();return base
    if args.stage!='test' and not (root/'frozen_membrain.pt').exists():
        paths={'fit':[],'validation':[],'calibration':[]}
        roles=[('fit',seed,384*i) for i,seed in enumerate(FIT_SEEDS)]+[
            ('validation',VALID_SEED,1152),('calibration',CAL_SEED,1536)]
        for role,seed,start in roles:
            base=base_for(seed)
            for corpus in SOURCES:
                x=data[corpus]
                for regime in ['standard','repeated']:
                    paths[role].append(cell(root,args.source,tok,base,x[start:start+384],x[2304:],corpus,seed,regime,role))
        models=freeze(root,paths['fit'],paths['validation'],paths['calibration'])
    else:
        if not (root/'frozen_membrain.pt').exists():raise RuntimeError('Complete development stage first')
        models=torch.load(root/'frozen_membrain.pt',weights_only=False)
    if args.stage=='development':return
    for seed in TARGET_SEEDS:
        base=base_for(seed)
        for corpus in SOURCES:
            x=data[corpus]
            for regime in ['standard','repeated']:
                path=cell(root,args.source,tok,base,x[1920:2304],x[2304:],corpus,seed,regime,'target')
                if not (path/'metrics.json').exists():evaluate(path,models)


if __name__=='__main__':main()
