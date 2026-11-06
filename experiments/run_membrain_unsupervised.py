"""Fresh benchmark of fixed Membrain scoring. Labels enter target training/metrics only."""
import argparse
import copy
import csv
import hashlib
import json
import re
import time
import urllib.request
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from run_spw_benchmark import SOURCES,make_neighbors,save_json
from spw_transformer import TinyTransformer,Oracle,train_lm
from spw_methods import spw_batch,evaluate_scores,zlib_score,lowercase_score,neighbor_score
from membrain_unsupervised import FRACTIONS,extract_curves,score_curves

SEEDS=[5101,5102,5103]
FRESH={
 'ag_news':dict(file='ag_news_test.csv',sha256='521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699',
  url='https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv'),
 'wikitext2':dict(file='wikitext2_test.txt',sha256='d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0',
  url='https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt')}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array_sha(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def documents(path,corpus,min_chars):
    if corpus=='ag_news':
        with path.open() as f:rows=[r[2] for r in csv.reader(f)]
    else:rows=[x for x in path.read_text().splitlines() if not x.strip().startswith('=')]
    rows=list(dict.fromkeys(re.sub(r'\s+',' ',x).strip() for x in rows))
    return [x for x in rows if len(x)>=min_chars]


def prepare(source,previous,root):
    tok=Tokenizer.from_file(str(source/'tokenizer.json'));seen=set()
    for archive in [source/'benchmark_data.npz',previous/'fresh_data.npz']:
        with np.load(archive) as z:
            for key in z.files:seen.update(map(tuple,z[key]))
    with np.load(source/'benchmark_data.npz') as z:background=z['background'].copy()
    for corpus,spec in SOURCES.items():
        path=source/'data'/spec['file']
        if sha(path)!=spec['sha256']:raise ValueError('Old public source checksum mismatch')
        docs=documents(path,corpus,300);order=np.random.default_rng(8341).permutation(len(docs))
        seen.update(tuple(tok.encode(docs[i]).ids[:48]) for i in order[:2048])
    exclusions=seen.copy();arrays={'background':background};manifest={'sources':FRESH,'corpora':{}}
    for corpus,spec in FRESH.items():
        path=root/'data'/spec['file']
        if sha(path)!=spec['sha256']:raise ValueError('Fresh public source checksum mismatch')
        docs=documents(path,corpus,200);order=np.random.default_rng(9257).permutation(len(docs));chunks=[];skipped=0
        for i in order:
            ids=tok.encode(docs[i]).ids[:48];key=tuple(ids)
            if len(ids)!=48 or key in seen:skipped+=1;continue
            seen.add(key);chunks.append(ids)
            if len(chunks)==384:break
        if len(chunks)!=384:raise ValueError(f'Insufficient fresh documents: {corpus}: {len(chunks)}')
        x=np.array(chunks,dtype=np.int64);arrays[corpus]=x
        assert not (set(map(tuple,x))&exclusions)
        manifest['corpora'][corpus]=dict(count=len(x),sha256=array_sha(x),skipped=skipped)
    np.savez_compressed(root/'benchmark_data.npz',**arrays)
    (root/'tokenizer.json').write_bytes((source/'tokenizer.json').read_bytes())
    save_json(root/'data_manifest.json',manifest)
    return tok,arrays


def freeze_protocol(root):
    directory=Path(__file__).parent
    names=['membrain_unsupervised.py','run_membrain_unsupervised.py','MEMBRAIN_UNSUPERVISED_PROTOCOL.md']
    hashes={name:sha(directory/name) for name in names}
    path=root/'frozen_protocol.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['hashes']!=hashes:raise ValueError('Code/protocol changed; use a new output directory')
    else:save_json(path,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,
           fitted_attack=False,membership_labels_used_for_scoring=False,score_is_probability=False))


def paired_bootstrap(y,scores,seed,count=1000):
    rng=np.random.default_rng(seed);pos=np.flatnonzero(y);neg=np.flatnonzero(1-y)
    draws=[np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))] for _ in range(count)]
    def auc(labels,scores):
        order=np.argsort(scores,kind='stable');s=scores[order];lab=labels[order]
        starts=np.r_[0,np.flatnonzero(np.diff(s))+1];ends=np.r_[starts[1:],len(s)]
        ranks=np.repeat((starts+ends+1)/2,ends-starts);n=labels.sum()
        return (ranks[lab==1].sum()-n*(n+1)/2)/(n*(len(labels)-n))
    values={name:np.array([auc(y[d],s[d]) for d in draws]) for name,s in scores.items()}
    point={name:evaluate_scores(y,s)['auc'] for name,s in scores.items()}
    return {name:dict(auc_ci95=np.quantile(v,[.025,.975]).tolist(),differences={
        ref:dict(delta=point[name]-point[ref],ci95=np.quantile(v-values[ref],[.025,.975]).tolist())
        for ref in ['LOSS','Min-k20%','SPW-MIA']}) for name,v in values.items()}


def run_cell(root,tok,base,proposal,pool,corpus,seed,regime):
    path=root/'cells'/f'{corpus}_{seed}_{regime}';path.mkdir(parents=True,exist_ok=True)
    if (path/'metrics.json').exists():return
    started=time.perf_counter();start_utc=datetime.now(timezone.utc).isoformat()
    order=np.random.default_rng(seed).permutation(384);members=order[:256];nonmembers=order[256:]
    eval_ids=np.r_[members[:128],nonmembers];test=pool[eval_ids];train=pool[members]
    epochs=4 if regime=='standard' else 12
    if regime=='repeated':train=np.r_[train,np.repeat(train[:32],7,axis=0)]
    model=copy.deepcopy(base);print('CELL',path.name,flush=True)
    history=train_lm(model,train,epochs,seed+71);torch.save(model.state_dict(),path/'target.pt')
    words=[tok.decode([i],skip_special_tokens=False) for i in range(tok.get_vocab_size())]
    oracle=Oracle(model,words);lp=oracle.teacher(test)
    token_lp=np.take_along_axis(lp,test[:,1:,None],axis=2)[:,:,0];nll=-token_lp.mean(1)
    details=spw_batch(oracle,test,teacher_log_probs=lp,return_token_details=True);del lp
    curves=extract_curves(model,test,details)
    # No labels, fitted weights, normalizer, calibration, or classifier enter here.
    scores=score_curves(curves,token_lp)
    scores['SPW-MIA']=np.array([d['spw'] for d in details])
    texts=[tok.decode(row.tolist()) for row in test]
    lower=oracle.nll([tok.encode(s.lower()).ids for s in texts])
    neighbors=make_neighbors(proposal,test,2871)
    neighbor_nll=oracle.nll(neighbors.reshape(-1,48)).reshape(len(test),10)
    scores.update(Zlib=np.array([zlib_score(a,b) for a,b in zip(nll,texts)]),
        Lowercase=np.array([lowercase_score(a,b) for a,b in zip(nll,lower)]),
        Neighbor=np.array([neighbor_score(a,b) for a,b in zip(nll,neighbor_nll)]))
    # Persist candidate-only scores before evaluating membership labels.
    np.savez_compressed(path/'unlabeled_scores.npz',eval_ids=eval_ids,**scores)
    y=np.r_[np.ones(128,dtype=int),np.zeros(128,dtype=int)]
    boot=paired_bootstrap(y,scores,seed);metrics={m:dict(**evaluate_scores(y,s),**boot[m]) for m,s in scores.items()}
    rng=np.random.default_rng(712);imbalance={}
    for gamma in [.05,.1,.2,.4,.8,1.]:
        draws=[np.r_[rng.choice(np.flatnonzero(y),max(1,round(gamma*128)),replace=False),np.flatnonzero(1-y)] for _ in range(20)]
        imbalance[str(gamma)]={m:float(np.mean([evaluate_scores(y[d],s[d])['average_precision'] for d in draws])) for m,s in scores.items()}
    np.savez_compressed(path/'scores.npz',labels=y,eval_ids=eval_ids,**scores)
    np.savez_compressed(path/'curves.npz',labels=y,**curves)
    save_json(path/'metrics.json',dict(corpus=corpus,seed=seed,regime=regime,started_utc=start_utc,
       history=history,metrics=metrics,imbalance=imbalance,seconds=time.perf_counter()-started,
       membership_assignment_sha256=array_sha(order),pool_sha256=array_sha(pool),oracle_stats=oracle.stats,
       unsupervised_attack=True,fitted_attack=False))
    print('RESULT',path.name,{m:round(v['auc'],4) for m,v in metrics.items()},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('results/spw_scale'))
    p.add_argument('--previous',type=Path,default=Path('results/membrain_prefix'))
    p.add_argument('--output',type=Path,default=Path('results/membrain_unsupervised'))
    p.add_argument('--download-data',action='store_true');p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    root=args.output;(root/'data').mkdir(parents=True,exist_ok=True)
    if args.download_data:
        for spec in FRESH.values():
            path=root/'data'/spec['file']
            if not path.exists():
                with urllib.request.urlopen(spec['url'],timeout=120) as response:payload=response.read()
                if hashlib.sha256(payload).hexdigest()!=spec['sha256']:raise ValueError('Download checksum mismatch')
                path.write_bytes(payload)
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    freeze_protocol(root);tok,data=prepare(args.source,args.previous,root)
    save_json(root/'config.json',dict(method='Membrain',attack='unsupervised fixed score',seeds=SEEDS,
        fractions=FRACTIONS,fit_stage=None,calibration=None,score_is_probability=False,
        sample_length=48,evaluation_members=128,evaluation_nonmembers=128,
        primary='Min-k20% + log(early-weighted recovery support)',bootstrap_draws=1000))
    proposal=TinyTransformer();proposal.load_state_dict(torch.load(args.source/'proposal.pt',weights_only=True));proposal.eval()
    for seed in SEEDS:
        torch.manual_seed(seed);base=TinyTransformer();path=root/f'base_{seed}.pt'
        if path.exists():base.load_state_dict(torch.load(path,weights_only=True))
        else:train_lm(base,data['background'],3,seed);torch.save(base.state_dict(),path)
        for corpus in FRESH:
            for regime in ['standard','repeated']:run_cell(root,tok,base,proposal,data[corpus],corpus,seed,regime)


if __name__=='__main__':main()
