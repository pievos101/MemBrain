"""Replay output-only Membrain from JSONL transcripts; no Torch/tokenizer needed."""
import argparse
import json
from pathlib import Path
from membrain_text import score_record,component_probability


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('transcripts',type=Path)
    p.add_argument('--mixture',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--word-probe',action='store_true',help='Replay the exploratory short-completion control')
    args=p.parse_args();scorer=score_record;primary='Membrain text'
    if args.word_probe:
        from membrain_word_probe import score_record as scorer
        primary='Membrain next-word probe'
    mixture=None
    if args.mixture:
        m=json.loads(args.mixture.read_text());mixture=m.get('model',m)
    with args.transcripts.open() as source,args.output.open('w') as output:
        for line in source:
            record=json.loads(line);scores,_=scorer(record)
            result=dict(text=record['text'],scores=scores,access='text strings only',membership_probability=None)
            if mixture:
                result['two_groups_selected']=mixture['selected_two']
                posterior=float(component_probability(mixture,[scores[primary]])[0])
                result['high_recovery_component_probability']=posterior
                result['group']=('high recovery' if posterior>=.5 else 'low recovery') if mixture['selected_two'] else 'abstain: one component preferred'
            output.write(json.dumps(result,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
