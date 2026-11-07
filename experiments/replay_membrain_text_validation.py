"""Replay validation scores/groups from strings only, without model libraries."""
import argparse
import json
from pathlib import Path
from membrain_text_validation import combined_scores
from membrain_text import component_probability


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cell',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    def read(name):
        with (args.cell/name).open() as f:return [json.loads(line) for line in f]
    first=read('primary_transcripts.jsonl');second=read('replicate_transcripts.jsonl');long=read('greedy_transcripts.jsonl')
    scores=combined_scores(first,second,long);models=json.loads((args.cell/'unlabeled_mixtures.json').read_text())
    model=models['models']['Membrain 16']['model'];posterior=component_probability(model,scores['Membrain 16'])
    with args.output.open('w') as f:
        for i,r in enumerate(first):
            f.write(json.dumps(dict(text=r['text'],scores={k:float(v[i]) for k,v in scores.items()},
                high_recovery_component_probability=float(posterior[i]),membership_probability=None,
                group=('high recovery' if posterior[i]>=.5 else 'low recovery') if model['selected_two'] else 'abstain: one component preferred'),ensure_ascii=False)+'\n')


if __name__=='__main__':main()
