"""Sentence benchmark owner: output-only attack, then privileged evaluation."""
import argparse
import copy
import json
import re
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from run_membrain_unsupervised import documents,sha,array_sha,FRESH,paired_bootstrap
from run_spw_benchmark import SOURCES,save_json
from spw_transformer import TinyTransformer,train_lm,Oracle
from spw_methods import spw_batch,evaluate_scores
from membrain_text import collect,score_record,fit_mixture,component_probability,words
from membrain_text_backend import TextCompletionService,train_sentences

SEEDS=[8101,8102,8103]
ARCHIVES=['results/spw_scale/benchmark_data.npz','results/membrain_prefix/fresh_data.npz',
 'results/membrain_unsupervised/benchmark_data.npz','results/membrain_simple/benchmark_data.npz','results/membrain_shape/benchmark_data.npz']


def prepare(root):
    source=Path('results/spw_scale');tok=Tokenizer.from_file(str(source/'tokenizer.json'));seen=set()
    for archive in ARCHIVES:
        with np.load(archive) as z:
            for key in z.files:seen.update(map(tuple,z[key]))
    with np.load(source/'benchmark_data.npz') as z:background=z['background'].copy()
    for c,spec in SOURCES.items():
        path=source/'data'/spec['file'];assert sha(path)==spec['sha256']
        docs=documents(path,c,300);order=np.random.default_rng(8341).permutation(len(docs))
        seen.update(tuple(tok.encode(docs[i]).ids[:48]) for i in order[:2048])
    texts={};manifest={};seen_sentences=set()
    for c,spec in FRESH.items():
        path=Path('results/membrain_unsupervised/data')/spec['file'];assert sha(path)==spec['sha256']
        docs=documents(path,c,200);selected=[];doc_ids=[]
        for i in np.random.default_rng(9257).permutation(len(docs)):
            doc=docs[i];key=tuple(tok.encode(doc).ids[:48])
            if key in seen:continue
            for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])',doc):
                canonical=' '.join(words(sentence));ids=tok.encode(sentence).ids
                if canonical in seen_sentences or not 12<=len(words(sentence))<=40 or not 16<=len(ids)<=80:continue
                if not re.search(r'[.!?]["’”\)\]]?$',sentence):continue
                seen.add(key);seen_sentences.add(canonical);selected.append(sentence);doc_ids.append(list(key));break
            if len(selected)==384:break
        if len(selected)!=384:raise ValueError(f'Need384 fresh sentences:{c}:{len(selected)}')
        texts[c]=selected;manifest[c]=dict(count=384,sentence_sha256=__import__('hashlib').sha256(json.dumps(selected).encode()).hexdigest(),
             source=spec,source_doc_chunks_sha256=array_sha(doc_ids))
        np.save(root/f'{c}_source_chunks.npy',np.array(doc_ids,dtype=np.int64))
    save_json(root/'sentences.json',texts);save_json(root/'data_manifest.json',manifest)
    (root/'tokenizer.json').write_bytes((source/'tokenizer.json').read_bytes());np.save(root/'background.npy',background)
    return tok,background,texts


def cluster_metrics(y,pred):
    if not len(y) or set(np.unique(y))!={0,1}:return None
    positive=pred==1;tp=int(np.sum(positive&(y==1)));fp=int(np.sum(positive&(y==0)))
    return dict(precision=tp/max(1,tp+fp),recall=tp/int(np.sum(y==1)),
        false_positive_rate=fp/int(np.sum(y==0)),balanced_accuracy=.5*(tp/int(np.sum(y==1))+1-fp/int(np.sum(y==0))),
        predicted_high=int(positive.sum()),count=len(y))


def privileged_references(model,tok,texts):
    """Owner-only comparison: NEVER inputs to output-only scores or mixtures."""
    sequences=[tok.encode(t).ids for t in texts];groups={};out={k:np.zeros(len(texts)) for k in ['LOSS','Min-k20%','SPW-MIA']}
    for i,row in enumerate(sequences):groups.setdefault(len(row),[]).append(i)
    oracle=Oracle(model,[tok.decode([i],skip_special_tokens=False) for i in range(tok.get_vocab_size())])
    for ids in groups.values():
        x=np.array([sequences[i] for i in ids]);lp=oracle.teacher(x)
        token_lp=np.take_along_axis(lp,x[:,1:,None],axis=2)[:,:,0]
        out['LOSS'][ids]=token_lp.mean(1);k=max(1,int(np.ceil(.2*token_lp.shape[1])))
        out['Min-k20%'][ids]=np.sort(token_lp,axis=1)[:,:k].mean(1)
        out['SPW-MIA'][ids]=[d['spw'] for d in spw_batch(oracle,x,teacher_log_probs=lp)]
    return out


def run_cell(root,tok,base,pool,corpus,seed,regime):
    path=root/'cells'/f'{corpus}_{seed}_{regime}';path.mkdir(parents=True,exist_ok=True)
    if (path/'metrics.json').exists():return
    start=datetime.now(timezone.utc).isoformat();order=np.random.default_rng(seed).permutation(384)
    ids=np.r_[order[:128],order[256:]];texts=[pool[i] for i in ids]
    train_ids=order[:256]
    if regime=='repeated':train_ids=np.r_[train_ids,np.repeat(train_ids[:32],7)]
    sequences=[tok.encode(pool[i]).ids+tok.encode('\n').ids for i in train_ids]
    model=copy.deepcopy(base);print('CELL',path.name,flush=True)
    history=train_sentences(model,sequences,4 if regime=='standard' else 12,seed+71)
    torch.save(model.state_dict(),path/'target.pt')
    service=TextCompletionService(model,tok);print('  output-only queries',len(texts),'sentences',flush=True)
    records=collect(texts,service,seed+517)
    with (path/'transcripts.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    evaluated=[score_record(r) for r in records];scores={k:np.array([s[k] for s,c in evaluated]) for k in evaluated[0][0]}
    save_json(path/'recovery_curves.json',[c for s,c in evaluated]);np.savez_compressed(path/'output_only_scores.npz',**scores)
    # Unlabeled discovery split and component orientation: no membership labels enter.
    shuffled=np.random.default_rng(8271).permutation(len(texts));discovery=shuffled[:128];heldout=shuffled[128:]
    mixture=fit_mixture(scores['Membrain text'][discovery]);posterior=component_probability(mixture,scores['Membrain text'][heldout])
    save_json(path/'unsupervised_mixture.json',dict(model=mixture,discovery=discovery.tolist(),heldout=heldout.tolist(),posterior=posterior.tolist()))
    print('  text scores saved; mixture frozen; computing privileged references',flush=True)
    # Labels appear below only for evaluation, after all attack outputs are fixed.
    y=np.r_[np.ones(128,dtype=int),np.zeros(128,dtype=int)];high=np.zeros(256,dtype=int)
    if regime=='repeated':high[:32]=1
    clustering=dict(selected_two=mixture['selected_two'],membership=None,high_exposure=None)
    if mixture['selected_two']:
        pred=(posterior>=.5).astype(int);clustering.update(membership=cluster_metrics(y[heldout],pred),high_exposure=cluster_metrics(high[heldout],pred))
    references=privileged_references(model,tok,texts);all_scores=dict(scores,**references)
    boot=paired_bootstrap(y,all_scores,seed);metrics={k:dict(**evaluate_scores(y,s),**boot[k]) for k,s in all_scores.items()}
    exposure_metrics={k:evaluate_scores(high,s) for k,s in scores.items()} if regime=='repeated' else None
    np.savez_compressed(path/'scores.npz',labels=y,high_exposure=high,eval_ids=ids,**all_scores)
    save_json(path/'metrics.json',dict(corpus=corpus,seed=seed,regime=regime,started_utc=start,history=history,
       metrics=metrics,high_exposure_metrics=exposure_metrics,clustering=clustering,query_stats=service.stats,
       membership_supervised=False,unsupervised_mixture=True,membership_assignment_sha256=array_sha(order)))
    print('RESULT',path.name,{k:round(v['auc'],4) for k,v in metrics.items()},'cluster',clustering,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/membrain_text'));args=p.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    directory=Path(__file__).parent
    hashes={n:sha(directory/n) for n in ['membrain_text.py','membrain_text_backend.py','run_membrain_text.py','MEMBRAIN_TEXT_PROTOCOL.md']}
    freeze=root/'frozen_protocol.json'
    if freeze.exists():
        if json.loads(freeze.read_text())['hashes']!=hashes:raise ValueError('Use fresh output after code/protocol changes')
    else:save_json(freeze,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,membership_supervised=False))
    tok,background,pools=prepare(root)
    save_json(root/'config.json',dict(method='Membrain text',seeds=SEEDS,attack_access='completion strings only',
         membership_supervised=False,grouping='unlabeled GMM discovery128, heldout128, BIC1 vs2',
         stochastic_samples=3,temperature=.8,maximum_new_tokens=96,whole_sentences=True))
    for seed in SEEDS:
        torch.manual_seed(seed);base=TinyTransformer();path=root/f'base_{seed}.pt'
        if path.exists():base.load_state_dict(torch.load(path,weights_only=True))
        else:train_lm(base,background,3,seed);torch.save(base.state_dict(),path)
        for corpus,pool in pools.items():
            for regime in ['standard','repeated']:run_cell(root,tok,base,pool,corpus,seed,regime)


if __name__=='__main__':main()
