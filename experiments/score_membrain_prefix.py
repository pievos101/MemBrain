"""Score a new text with a trusted local Membrain attack and compatible target."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from run_spw_benchmark import FRACTIONS
from spw_transformer import TinyTransformer,Oracle
from spw_methods import spw_batch
from membrain_prefix import recovery_features,assemble_features,decision,probabilities


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--attack-dir',type=Path,default=Path('results/membrain_prefix'))
    p.add_argument('--target-checkpoint',type=Path,required=True)
    p.add_argument('--text-file',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(args.threads)
    tok=Tokenizer.from_file(str(args.attack_dir/'tokenizer.json'))
    ids=tok.encode(args.text_file.read_text().strip()).ids
    if len(ids)<48:raise ValueError('This benchmark attack requires at least 48 BPE tokens of input.')
    x=np.asarray([ids[:48]])
    model=TinyTransformer();model.load_state_dict(torch.load(args.target_checkpoint,weights_only=True));model.eval()
    oracle=Oracle(model,[tok.decode([i],skip_special_tokens=False) for i in range(tok.get_vocab_size())])
    lp=oracle.teacher(x);token_lp=np.take_along_axis(lp,x[:,1:,None],axis=2)[:,:,0]
    details=spw_batch(oracle,x,teacher_log_probs=lp,return_token_details=True)
    curves,_,_=recovery_features(model,x,details,FRACTIONS)
    features,names,groups=assemble_features(curves,details,token_lp,FRACTIONS)
    schema=json.loads((args.attack_dir/'feature_schema.json').read_text())
    if dict(names=names,groups=groups)!=schema:raise ValueError('Incompatible feature schema')
    # Our own local checkpoint contains NumPy metadata; never load an untrusted pickle here.
    attack=torch.load(args.attack_dir/'frozen_membrain.pt',weights_only=False)['Membrain']
    logit=decision(attack,features[:,attack['mask']]);probability=float(probabilities(attack['calibration'],logit)[0])
    print(json.dumps(dict(method='Membrain',membership_score=probability,
        probability_assumptions='Balanced membership prior and matching shadow/target training distribution; not validated for arbitrary external models.',
        calibration_threshold=attack['threshold'],above_calibration_threshold=probability>=attack['threshold'],
        evaluated_tokens=48,input_tokens=len(ids),truncated=len(ids)>48,prefix_fractions=FRACTIONS,
        curves={k:v[0].tolist() for k,v in curves.items()}),indent=2))


if __name__=='__main__':main()
