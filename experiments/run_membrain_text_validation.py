"""Fresh, frozen validation of output-only next-word recovery and grouping."""
import argparse
import copy
import hashlib
import json
import re
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from run_membrain_unsupervised import documents,sha,array_sha,paired_bootstrap
from run_spw_benchmark import SOURCES,save_json
from run_membrain_text import ARCHIVES,cluster_metrics,privileged_references
from spw_transformer import TinyTransformer,train_lm
from membrain_text_backend import TextCompletionService,train_sentences
from membrain_text import words,fit_mixture,component_probability
from membrain_text_validation import collect,combined_scores,rank_correlation
from spw_methods import evaluate_scores

SEEDS=[9101,9102]
CAPACITIES={'small':dict(d=64,layers=2),'larger':dict(d=96,layers=3)}
VALID_SHA='f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8'


def save_jsonl(path,records):
    with path.open('w') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def prepare(root):
    tok=Tokenizer.from_file('results/spw_scale/tokenizer.json');seen=set()
    for archive in ARCHIVES:
        with np.load(archive) as z:
            for key in z.files:seen.update(map(tuple,z[key]))
    for corpus in SOURCES:seen.update(map(tuple,np.load(f'results/membrain_text/{corpus}_source_chunks.npy')))
    prior=json.loads(Path('results/membrain_text/sentences.json').read_text())
    canonical_seen={' '.join(words(t)) for pool in prior.values() for t in pool}
    with np.load('results/spw_scale/benchmark_data.npz') as z:background=z['background'].copy()
    for corpus,spec in SOURCES.items():
        path=Path('results/spw_scale/data')/spec['file'];assert sha(path)==spec['sha256']
        docs=documents(path,corpus,300);order=np.random.default_rng(8341).permutation(len(docs))
        seen.update(tuple(tok.encode(docs[i]).ids[:48]) for i in order[:2048])
    paths={'ag_news':Path('results/spw_scale/data/ag_news_train.csv'),'wikitext2':root/'data/wikitext2_valid.txt'}
    assert sha(paths['ag_news'])==SOURCES['ag_news']['sha256'];assert sha(paths['wikitext2'])==VALID_SHA
    pools={};manifest={}
    for corpus,path in paths.items():
        docs=documents(path,corpus,200);selected=[];chunks=[]
        for i in np.random.default_rng(44291).permutation(len(docs)):
            key=tuple(tok.encode(docs[i]).ids[:48])
            if len(key)!=48 or key in seen:continue
            for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])',docs[i]):
                canonical=' '.join(words(sentence));encoded=tok.encode(sentence).ids
                if canonical in canonical_seen or not 12<=len(words(sentence))<=40 or not 16<=len(encoded)<=80:continue
                if not re.search(r'[.!?]["’”\)\]]?$',sentence):continue
                seen.add(key);canonical_seen.add(canonical);selected.append(sentence);chunks.append(key);break
            if len(selected)==768:break
        if len(selected)!=768:raise ValueError(f'Insufficient fresh sentences:{corpus}:{len(selected)}')
        pools[corpus]=selected;np.save(root/f'{corpus}_source_chunks.npy',np.array(chunks))
        manifest[corpus]=dict(count=768,source_path=str(path),source_sha256=sha(path),document_chunks_sha256=array_sha(chunks),
           pool_sha256=[hashlib.sha256(json.dumps(selected[i:i+384]).encode()).hexdigest() for i in [0,384]])
    save_json(root/'sentences.json',pools);save_json(root/'data_manifest.json',manifest)
    np.save(root/'background.npy',background);(root/'tokenizer.json').write_bytes(Path('results/spw_scale/tokenizer.json').read_bytes())
    return tok,background,pools


def fit_groups(path,scores,discovery,heldout):
    result={}
    for name in ['Membrain 16','Membrain replicate16','Membrain pooled32']:
        model=fit_mixture(scores[name][discovery]);p=component_probability(model,scores[name][heldout])
        result[name]=dict(model=model,posterior=p.tolist())
    save_json(path/'unlabeled_mixtures.json',dict(discovery=discovery.tolist(),heldout=heldout.tolist(),models=result))
    return result


def run_cell(root,tok,base,pool,corpus,capacity,seed,pool_id,regime):
    path=root/'cells'/f'{capacity}_{corpus}_{seed}_{regime}';path.mkdir(parents=True,exist_ok=True)
    if (path/'metrics.json').exists():return
    started=datetime.now(timezone.utc).isoformat();order=np.random.default_rng(seed).permutation(384)
    members=order[:256];ids=np.r_[members[:128],order[256:]]
    blind_order=np.random.default_rng(seed+11417).permutation(256)
    ids=ids[blind_order];texts=[pool[i] for i in ids]
    train_ids=members if regime=='standard' else np.r_[members,np.repeat(members[:32],7)]
    sequences=[tok.encode(pool[i]).ids+tok.encode('\n').ids for i in train_ids]
    model=copy.deepcopy(base);print('CELL',path.name,flush=True)
    history=train_sentences(model,sequences,4 if regime=='standard' else 12,seed+71);torch.save(model.state_dict(),path/'target.pt')
    service=TextCompletionService(model,tok)
    print('  primary16 short completions',flush=True)
    first=collect(texts,service,16,8,seed+9517);save_jsonl(path/'primary_transcripts.jsonl',first)
    print('  independent16 short completions',flush=True)
    replicate=collect(texts,service,16,8,seed+19517);save_jsonl(path/'replicate_transcripts.jsonl',replicate)
    print('  greedy full-completion control',flush=True)
    long=collect(texts,service,0,96,seed+517);save_jsonl(path/'greedy_transcripts.jsonl',long)
    scores=combined_scores(first,replicate,long);np.savez_compressed(path/'output_only_scores.npz',**scores)
    shuffled=np.random.default_rng(8271).permutation(256);discovery=shuffled[:128];heldout=shuffled[128:]
    groups=fit_groups(path,scores,discovery,heldout)
    # All attack outputs and unlabeled fits are saved before owner-side evaluation.
    y=np.r_[np.ones(128,dtype=int),np.zeros(128,dtype=int)][blind_order]
    high=np.r_[np.ones(32,dtype=int),np.zeros(224,dtype=int)][blind_order] if regime=='repeated' else np.zeros(256,dtype=int)
    clustering={}
    for name,g in groups.items():
        c=dict(selected_two=g['model']['selected_two'],membership=None,high_exposure=None,
               heldout_members=int(y[heldout].sum()),heldout_nonmembers=int((1-y[heldout]).sum()))
        if c['selected_two']:
            prediction=(np.array(g['posterior'])>=.5).astype(int)
            c.update(membership=cluster_metrics(y[heldout],prediction),high_exposure=cluster_metrics(high[heldout],prediction))
        clustering[name]=c
    stability=dict(spearman=rank_correlation(scores['Membrain 16'],scores['Membrain replicate16']),high_group_jaccard=None)
    if all(groups[n]['model']['selected_two'] for n in ['Membrain 16','Membrain replicate16']):
        a=np.array(groups['Membrain 16']['posterior'])>=.5;b=np.array(groups['Membrain replicate16']['posterior'])>=.5
        stability['high_group_jaccard']=float((a&b).sum()/max(1,(a|b).sum()))
    print('  score/group freeze complete; privileged reference evaluation',flush=True)
    refs=privileged_references(model,tok,texts);all_scores=dict(scores,**refs);boot=paired_bootstrap(y,all_scores,seed)
    swapped=dict(all_scores);swapped['Raw LOSS']=all_scores['LOSS'];swapped['LOSS']=all_scores['Greedy whole-suffix']
    greedy_boot=paired_bootstrap(y,swapped,seed)
    for name in boot:
        key='Raw LOSS' if name=='LOSS' else name
        boot[name]['differences']['Greedy whole-suffix']=greedy_boot[key]['differences']['LOSS']
    metrics={n:dict(**evaluate_scores(y,s),**boot[n]) for n,s in all_scores.items()}
    exposure={n:evaluate_scores(high,s) for n,s in scores.items()} if regime=='repeated' else None
    np.savez_compressed(path/'scores.npz',labels=y,high_exposure=high,eval_ids=ids,**all_scores)
    save_json(path/'metrics.json',dict(capacity=capacity,corpus=corpus,seed=seed,pool_id=pool_id,regime=regime,
       started_utc=started,history=history,parameter_count=sum(p.numel() for p in model.parameters()),metrics=metrics,
       clustering=clustering,stability=stability,high_exposure_metrics=exposure,query_stats=service.stats,
       membership_supervised=False,assignment_sha256=array_sha(order),blind_query_order_sha256=array_sha(blind_order)))
    print('RESULT',path.name,{n:round(v['auc'],4) for n,v in metrics.items()},'stability',stability,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/membrain_text_validation'));args=p.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    directory=Path(__file__).parent
    names=['membrain_word_probe.py','membrain_text.py','membrain_text_backend.py','membrain_text_validation.py',
           'run_membrain_text_validation.py','MEMBRAIN_TEXT_VALIDATION.md']
    hashes={n:sha(directory/n) for n in names};frozen=root/'frozen_protocol.json'
    old=json.loads(Path('results/membrain_text/word_probe_freeze.json').read_text())
    if hashes['membrain_word_probe.py']!=old['hashes']['membrain_word_probe.py']:raise ValueError('Development score changed')
    if frozen.exists():
        if json.loads(frozen.read_text())['hashes']!=hashes:raise ValueError('Use fresh output after frozen code changes')
    else:save_json(frozen,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,prospective_internal_validation=True,
                              development_commit='47c7358',membership_supervised=False))
    tok,background,pools=prepare(root)
    save_json(root/'config.json',dict(primary='Membrain 16',seeds=SEEDS,capacities=CAPACITIES,independent_pools_per_corpus=2,
        sample_counts=[3,16,16,32],temperature=.8,short_cap=8,whole_cap=96,membership_supervised=False))
    for capacity,kw in CAPACITIES.items():
        for pool_id,seed in enumerate(SEEDS):
            torch.manual_seed(seed);base=TinyTransformer(**kw);path=root/f'base_{capacity}_{seed}.pt'
            if path.exists():base.load_state_dict(torch.load(path,weights_only=True))
            else:train_lm(base,background,3,seed);torch.save(base.state_dict(),path)
            for corpus,texts in pools.items():
                pool=texts[pool_id*384:(pool_id+1)*384]
                for regime in ['standard','repeated']:run_cell(root,tok,base,pool,corpus,capacity,seed,pool_id,regime)


if __name__=='__main__':main()
