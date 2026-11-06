"""Fresh test of two-trajectory Membrain; scoring has no membership-label input."""
import copy
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from run_membrain_unsupervised import documents,sha,array_sha,FRESH,paired_bootstrap
from run_spw_benchmark import SOURCES,make_neighbors,save_json
from spw_transformer import TinyTransformer,Oracle,train_lm
from spw_methods import spw_batch,evaluate_scores,zlib_score,lowercase_score,neighbor_score
from membrain_unsupervised import extract_curves as old_extract,score_curves as old_score
from membrain_simple import extract,score

ROOT=Path('results/membrain_simple');SEEDS=[6101,6102,6103]


def prepare():
    source=Path('results/spw_scale');previous=Path('results/membrain_unsupervised')
    tok=Tokenizer.from_file(str(source/'tokenizer.json'));seen=set()
    for archive in [source/'benchmark_data.npz',Path('results/membrain_prefix/fresh_data.npz'),previous/'benchmark_data.npz']:
        with np.load(archive) as z:
            for key in z.files:seen.update(map(tuple,z[key]))
    with np.load(source/'benchmark_data.npz') as z:data={'background':z['background'].copy()}
    for corpus,spec in SOURCES.items():
        path=source/'data'/spec['file'];assert sha(path)==spec['sha256']
        docs=documents(path,corpus,300);order=np.random.default_rng(8341).permutation(len(docs))
        seen.update(tuple(tok.encode(docs[i]).ids[:48]) for i in order[:2048])
    manifest={}
    for corpus,spec in FRESH.items():
        path=previous/'data'/spec['file'];assert sha(path)==spec['sha256']
        docs=documents(path,corpus,200);chunks=[]
        for i in np.random.default_rng(9257).permutation(len(docs)):
            ids=tok.encode(docs[i]).ids[:48];key=tuple(ids)
            if len(ids)!=48 or key in seen:continue
            seen.add(key);chunks.append(ids)
            if len(chunks)==384:break
        if len(chunks)!=384:raise ValueError(f'Insufficient fresh chunks:{corpus}:{len(chunks)}')
        data[corpus]=np.array(chunks);manifest[corpus]=dict(count=384,sha256=array_sha(data[corpus]),source=spec)
    np.savez_compressed(ROOT/'benchmark_data.npz',**data)
    (ROOT/'tokenizer.json').write_bytes((source/'tokenizer.json').read_bytes());save_json(ROOT/'data_manifest.json',manifest)
    return tok,data


def cell(tok,base,proposal,pool,corpus,seed,regime):
    path=ROOT/'cells'/f'{corpus}_{seed}_{regime}';path.mkdir(parents=True,exist_ok=True)
    if (path/'metrics.json').exists():return
    start=datetime.now(timezone.utc).isoformat();order=np.random.default_rng(seed).permutation(384)
    ids=np.r_[order[:128],order[256:]];test=pool[ids];train=pool[order[:256]]
    if regime=='repeated':train=np.r_[train,np.repeat(train[:32],7,axis=0)]
    model=copy.deepcopy(base);print('CELL',path.name,flush=True)
    history=train_lm(model,train,4 if regime=='standard' else 12,seed+71);torch.save(model.state_dict(),path/'target.pt')
    curves=extract(model,test);scores=score(curves)
    oracle=Oracle(model,[tok.decode([i],skip_special_tokens=False) for i in range(tok.get_vocab_size())])
    lp=oracle.teacher(test);token_lp=np.take_along_axis(lp,test[:,1:,None],axis=2)[:,:,0];nll=-token_lp.mean(1)
    detail=spw_batch(oracle,test,teacher_log_probs=lp,return_token_details=True);del lp
    previous=old_score(old_extract(model,test,detail),token_lp)
    scores.update({'Previous Membrain':previous['Membrain'],'LOSS':previous['LOSS'],'Min-k20%':previous['Min-k20%'],
                   'SPW-MIA':np.array([d['spw'] for d in detail])})
    texts=[tok.decode(row.tolist()) for row in test];lower=oracle.nll([tok.encode(s.lower()).ids for s in texts])
    neighbor_tokens=make_neighbors(proposal,test,2871);neighbor=oracle.nll(neighbor_tokens.reshape(-1,48)).reshape(len(test),10)
    scores.update(Zlib=np.array([zlib_score(a,b) for a,b in zip(nll,texts)]),
        Lowercase=np.array([lowercase_score(a,b) for a,b in zip(nll,lower)]),
        Neighbor=np.array([neighbor_score(a,b) for a,b in zip(nll,neighbor)]))
    np.savez_compressed(path/'unlabeled_scores.npz',eval_ids=ids,**scores)
    y=np.r_[np.ones(128,dtype=int),np.zeros(128,dtype=int)]
    boot=paired_bootstrap(y,scores,seed)
    # Same paired draws, adding the indispensable teacher-only control as reference.
    swapped=dict(scores);swapped['Raw LOSS']=scores['LOSS'];swapped['LOSS']=scores['Teacher loss curve']
    teacher_boot=paired_bootstrap(y,swapped,seed)
    for m in boot:
        key='Raw LOSS' if m=='LOSS' else m
        boot[m]['differences']['Teacher loss curve']=teacher_boot[key]['differences']['LOSS']
    metrics={m:dict(**evaluate_scores(y,s),**boot[m]) for m,s in scores.items()}
    np.savez_compressed(path/'scores.npz',labels=y,eval_ids=ids,**scores)
    np.savez_compressed(path/'trajectories.npz',labels=y,**curves)
    save_json(path/'metrics.json',dict(corpus=corpus,seed=seed,regime=regime,started_utc=start,history=history,
        metrics=metrics,fitted_attack=False,pool_sha256=array_sha(pool),membership_assignment_sha256=array_sha(order)))
    print('RESULT',path.name,{m:round(v['auc'],4) for m,v in metrics.items()},flush=True)


def main():
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);args=p.parse_args();ROOT=args.output
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    directory=Path(__file__).parent
    hashes={name:sha(directory/name) for name in ['membrain_simple.py','run_membrain_simple.py','MEMBRAIN_SIMPLE_PROTOCOL.md']}
    frozen=ROOT/'frozen_protocol.json'
    if frozen.exists():
        if json.loads(frozen.read_text())['hashes']!=hashes:raise ValueError('Use fresh output after code changes')
    else:save_json(frozen,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,fitted_attack=False))
    tok,data=prepare();proposal=TinyTransformer()
    proposal.load_state_dict(torch.load('results/spw_scale/proposal.pt',weights_only=True));proposal.eval()
    save_json(ROOT/'config.json',dict(method='Membrain',primary='-A[teacher CE + embedding distance * absolute rollout CE]',
         seeds=SEEDS,attack_supervision=None,score_is_probability=False))
    for seed in SEEDS:
        torch.manual_seed(seed);base=TinyTransformer();path=ROOT/f'base_{seed}.pt'
        if path.exists():base.load_state_dict(torch.load(path,weights_only=True))
        else:train_lm(base,data['background'],3,seed);torch.save(base.state_dict(),path)
        for corpus in FRESH:
            for regime in ['standard','repeated']:cell(tok,base,proposal,data[corpus],corpus,seed,regime)


if __name__=='__main__':main()
