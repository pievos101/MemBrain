"""Exploratory short-completion control using text-only scoring and unlabeled fit."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from tokenizers import Tokenizer
from membrain_text import fit_mixture,component_probability
from membrain_word_probe import score_record
from membrain_text_backend import TextCompletionService
from spw_transformer import TinyTransformer
from spw_methods import evaluate_scores
from run_membrain_unsupervised import sha
from run_membrain_text import cluster_metrics


def main():
    root=Path('results/membrain_text');torch.set_num_threads(4)
    directory=Path(__file__).parent;freeze=root/'word_probe_freeze.json'
    hashes={n:sha(directory/n) for n in ['membrain_word_probe.py','run_membrain_word_probe.py','MEMBRAIN_WORD_PROBE_CONTROL.md']}
    if freeze.exists():
        if json.loads(freeze.read_text())['hashes']!=hashes:raise ValueError('Control source changed')
    else:freeze.write_text(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,exploratory=True),indent=2)+'\n')
    tok=Tokenizer.from_file(str(root/'tokenizer.json'));paths=sorted((root/'cells').glob('*/metrics.json'))
    if len(paths)!=12:raise RuntimeError('Finish main study first')
    for p in paths:
        out=p.parent/'word_probe_metrics.json'
        if out.exists():continue
        with (p.parent/'transcripts.jsonl').open() as f:records=[json.loads(line) for line in f]
        prefixes=[q['prefix'] for r in records for q in r['queries']]
        model=TinyTransformer();model.load_state_dict(torch.load(p.parent/'target.pt',weights_only=True));service=TextCompletionService(model,tok)
        info=json.loads(p.read_text());print('WORD PROBE',p.parent.name,flush=True)
        answers=service(prefixes,samples=16,temperature=.8,max_new_tokens=8,seed=info['seed']+9517)
        at=0;new=[]
        for r in records:
            queries=[]
            for q in r['queries']:
                queries.append(dict(prefix=q['prefix'],suffix=q['suffix'],progress=q['progress'],answer=answers[at]));at+=1
            new.append(dict(text=r['text'],queries=queries))
        with (p.parent/'word_probe_transcripts.jsonl').open('w') as f:
            for r in new:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        evaluated=[score_record(r) for r in new];scores={k:np.array([s[k] for s,c in evaluated]) for k in evaluated[0][0]}
        np.savez_compressed(p.parent/'word_probe_unlabeled_scores.npz',**scores)
        split=json.loads((p.parent/'unsupervised_mixture.json').read_text());disc=np.array(split['discovery']);test=np.array(split['heldout'])
        mixture=fit_mixture(scores['Membrain next-word probe'][disc]);posterior=component_probability(mixture,scores['Membrain next-word probe'][test])
        (p.parent/'word_probe_mixture.json').write_text(json.dumps(dict(model=mixture,discovery=disc.tolist(),heldout=test.tolist(),posterior=posterior.tolist()),indent=2)+'\n')
        with np.load(p.parent/'scores.npz') as z:y=z['labels'];high=z['high_exposure']
        grouping=dict(selected_two=mixture['selected_two'],membership=None)
        if mixture['selected_two']:grouping['membership']=cluster_metrics(y[test],(posterior>=.5).astype(int))
        result=dict(exploratory=True,metrics={k:evaluate_scores(y,s) for k,s in scores.items()},
            high_exposure_metrics={k:evaluate_scores(high,s) for k,s in scores.items()} if info['regime']=='repeated' else None,
            clustering=grouping,query_stats=service.stats)
        out.write_text(json.dumps(result,indent=2)+'\n')
        print('PROBE RESULT',p.parent.name,{k:round(v['auc'],4) for k,v in result['metrics'].items()},flush=True)


if __name__=='__main__':main()
