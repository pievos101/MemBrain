"""Score text with two-trajectory Membrain. Only compatible target/tokenizer needed."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from spw_transformer import TinyTransformer
from membrain_simple import extract,score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer',type=Path,default=Path('results/membrain_simple/tokenizer.json'))
    p.add_argument('--target-checkpoint',type=Path,required=True);p.add_argument('--text-file',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    tok=Tokenizer.from_file(str(args.tokenizer));text=re.sub(r'\s+',' ',args.text_file.read_text()).strip();ids=tok.encode(text).ids
    if len(ids)<48:raise ValueError('This small-model implementation requires at least48 BPE tokens')
    model=TinyTransformer();model.load_state_dict(torch.load(args.target_checkpoint,weights_only=True));model.eval()
    curves=extract(model,np.array([ids[:48]]));scores=score(curves)
    print(json.dumps(dict(method='Membrain',fitted_attack=False,score_is_probability=False,
        score=float(scores['Membrain'][0]),evaluated_tokens=48,input_tokens=len(ids),
        curves={k:v[0].tolist() for k,v in curves.items()},comparison_scores={k:float(v[0]) for k,v in scores.items()}),indent=2))


if __name__=='__main__':main()
