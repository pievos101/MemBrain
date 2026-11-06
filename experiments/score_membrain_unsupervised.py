"""Score a candidate using Membrain's fixed formula; no attack checkpoint needed."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from spw_transformer import TinyTransformer,Oracle
from spw_methods import spw_batch
from membrain_unsupervised import FRACTIONS,extract_curves,score_curves


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer',type=Path,default=Path('results/membrain_unsupervised/tokenizer.json'))
    p.add_argument('--target-checkpoint',type=Path,required=True)
    p.add_argument('--text-file',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    torch.set_num_threads(args.threads);tok=Tokenizer.from_file(str(args.tokenizer))
    text=re.sub(r'\s+',' ',args.text_file.read_text()).strip();ids=tok.encode(text).ids
    if len(ids)<48:raise ValueError('This benchmark implementation requires at least 48 BPE tokens')
    x=np.asarray([ids[:48]]);model=TinyTransformer()
    model.load_state_dict(torch.load(args.target_checkpoint,weights_only=True));model.eval()
    oracle=Oracle(model,[tok.decode([i],skip_special_tokens=False) for i in range(tok.get_vocab_size())])
    lp=oracle.teacher(x);token_lp=np.take_along_axis(lp,x[:,1:,None],axis=2)[:,:,0]
    details=spw_batch(oracle,x,teacher_log_probs=lp,return_token_details=True)
    curves=extract_curves(model,x,details);scores=score_curves(curves,token_lp)
    print(json.dumps(dict(method='Membrain',unsupervised=True,fitted_attack=False,
      score=float(scores['Membrain'][0]),score_interpretation='Ranking score; higher means stronger membership evidence. Not a membership probability.',
      evaluated_tokens=48,input_tokens=len(ids),truncated=len(ids)>48,prefix_fractions=FRACTIONS,
      comparison_scores={k:float(v[0]) for k,v in scores.items()},curves={k:v[0].tolist() for k,v in curves.items()}),indent=2))


if __name__=='__main__':main()
