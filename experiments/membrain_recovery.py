"""Membrain: whole-sentence recovery as the supplied prefix increases.

Four Latin-rotation models expose every evaluation sentence at every repetition
level once. These are natural Shakespeare sentences, not Markov sequences.
"""
import argparse
import copy
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import membrain_loss_curves as previous

LEVELS=np.array([0,1,4,16])
FRACTIONS=np.r_[.01,.02,.05,np.arange(1,10)/10]
ALPHABET=''.join(chr(i) for i in range(32,127))
VOCAB=96  # 0 is PAD, printable ASCII maps to 1..95.
TAIL=8


def sentence_pool():
    data=previous.SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest()==previous.SOURCE_SHA
    text=re.sub(r'\s+',' ',data.decode('utf-8'))
    sentences=[];seen=set()
    for part in re.split(r'(?<=[.!?])\s+',text):
        # Remove speaker headers rather than treating them as sentence content.
        part=re.sub(r'^[A-Za-z][A-Za-z ]{0,35}:\s*','',part).strip()
        part=''.join(c if c in ALPHABET else ' ' for c in part)
        part=re.sub(r'\s+',' ',part)
        if 80<=len(part)<=120 and part[-1:] in '.!?' and part not in seen:
            seen.add(part);sentences.append(part)
    assert len(sentences)>=256,len(sentences)
    rng=np.random.default_rng(61721)
    ids=rng.choice(len(sentences),256,replace=False)
    return [sentences[i] for i in ids[:128]],[sentences[i] for i in ids[128:]]


def encode(texts):
    lengths=torch.tensor([len(t) for t in texts])
    x=torch.zeros(len(texts),int(lengths.max()),dtype=torch.long)
    for i,t in enumerate(texts):x[i,:len(t)]=torch.tensor([ord(c)-31 for c in t])
    return x,lengths


class SentenceLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(VOCAB,32,padding_idx=0)
        self.gru=nn.GRU(32,96,batch_first=True)
        self.output=nn.Linear(96,VOCAB)

    def forward(self,x):
        h,_=self.gru(self.embedding(x));return self.output(h)


def train(lm,x,repeats,epochs,lr=.003):
    indices=torch.repeat_interleave(torch.arange(len(x)),torch.tensor(repeats))
    assert len(indices)>0
    opt=torch.optim.Adam(lm.parameters(),lr=lr)
    history=[];lm.train()
    for epoch in range(epochs):
        values=[]
        for order in torch.randperm(len(indices)).split(32):
            b=x[indices[order]]
            loss=F.cross_entropy(lm(b[:,:-1]).reshape(-1,VOCAB),b[:,1:].reshape(-1),ignore_index=0)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(lm.parameters(),1.);opt.step()
            values.append(loss.item())
        history.append(float(np.mean(values)))
        if (epoch+1)%10==0:print(f'  epoch {epoch+1}/{epochs}: loss={history[-1]:.3f}',flush=True)
    return history


@torch.no_grad()
def recover(lm,texts,samples,return_embeddings=False):
    """Generate the WHOLE remaining sentence from each true prefix.

    Path 0 is greedy; remaining paths sample at temperature 1. Prefixes are
    encoded with packed sequences so padding never enters their hidden state.
    True suffix tokens are targets only, never generation inputs.
    """
    x,lengths=encode(texts);lm.eval()
    teacher_h,_=lm.gru(lm.embedding(x))
    teacher_tokens=F.cross_entropy(lm(x[:,:-1]).transpose(1,2),x[:,1:],ignore_index=0,reduction='none')
    metrics={k:[] for k in ['suffix_exact','suffix_accuracy','tail_exact','tail_accuracy',
                            'suffix_cosine','tail_cosine','rollout_ce','tail_rollout_ce']}
    teacher_ce=[];prefix_sizes=[];greedy_texts=[];original_rows=[];generated_rows=[]
    for fraction in FRACTIONS:
        prefix=torch.round(lengths.float()*float(fraction)).long().clamp_min(1)
        remaining=lengths-prefix
        assert int(remaining.min())>=TAIL
        prefix_sizes.append(prefix.numpy())
        mask=torch.arange(x.shape[1])[None,:]>=prefix[:,None]
        mask&=torch.arange(x.shape[1])[None,:]<lengths[:,None]
        original=(teacher_h*mask[:,:,None]).sum(1)/remaining[:,None]
        tailmask=(torch.arange(x.shape[1])[None,:]>=lengths[:,None]-TAIL)&(x!=0)
        original_tail=(teacher_h*tailmask[:,:,None]).sum(1)/TAIL
        tmask=(torch.arange(x.shape[1]-1)[None,:]>=prefix[:,None]-1)&(torch.arange(x.shape[1]-1)[None,:]<lengths[:,None]-1)
        teacher_ce.append((teacher_tokens*tmask).sum(1).numpy()/remaining.numpy())
        npaths=samples+1
        repeated=x.repeat_interleave(npaths,0)
        p=prefix.repeat_interleave(npaths);rem=remaining.repeat_interleave(npaths)
        lens=lengths.repeat_interleave(npaths)
        packed=nn.utils.rnn.pack_padded_sequence(lm.embedding(repeated[:,:int(p.max())]),p.cpu(),batch_first=True,enforce_sorted=False)
        _,state=lm.gru(packed)
        generated=[];hidden=[];losses=[]
        for j in range(int(rem.max())):
            logits=lm.output(state[-1])
            # PAD is never a legal generated text character. Keep raw logits for CE.
            sampling=logits.clone();sampling[:,0]=-torch.inf
            token=torch.multinomial(sampling.softmax(-1),1)
            greedy_rows=torch.arange(len(token))%npaths==0
            token[greedy_rows]=sampling[greedy_rows].argmax(-1,keepdim=True)
            target=repeated.gather(1,(p+j).clamp_max(repeated.shape[1]-1)[:,None])[:,0]
            losses.append(-logits.log_softmax(-1).gather(1,target[:,None])[:,0])
            h,state=lm.gru(lm.embedding(token),state)
            generated.append(token[:,0]);hidden.append(h[:,0])
        generated=torch.stack(generated,1);hidden=torch.stack(hidden,1);ce=torch.stack(losses,1)
        positions=p[:,None]+torch.arange(generated.shape[1])[None,:]
        active=torch.arange(generated.shape[1])[None,:]<rem[:,None]
        tail=(positions>=lens[:,None]-TAIL)&active
        target=repeated.gather(1,positions.clamp_max(repeated.shape[1]-1))
        match=generated==target
        hmean=(hidden*active[:,:,None]).sum(1)/rem[:,None]
        htail=(hidden*tail[:,:,None]).sum(1)/TAIL
        if return_embeddings:
            original_rows.append(original.numpy())
            generated_rows.append(hmean.reshape(len(texts),npaths,96).numpy())
        values=dict(suffix_exact=(match|~active).all(1).float(),
                    suffix_accuracy=(match&active).sum(1)/rem,
                    tail_exact=(match|~tail).all(1).float(),
                    tail_accuracy=(match&tail).sum(1)/TAIL,
                    suffix_cosine=F.cosine_similarity(hmean,original.repeat_interleave(npaths,0)),
                    tail_cosine=F.cosine_similarity(htail,original_tail.repeat_interleave(npaths,0)),
                    rollout_ce=(ce*active).sum(1)/rem,
                    tail_rollout_ce=(ce*tail).sum(1)/TAIL)
        for key,value in values.items():metrics[key].append(value.reshape(len(texts),npaths).numpy())
        # All examples are retained, not selected after viewing their success.
        greedy_texts.append([''.join(chr(int(t)+31) for t in generated[i*npaths,:remaining[i]]) for i in range(len(texts))])
    result={k:np.stack(v,axis=1) for k,v in metrics.items()}
    result.update(prefix_sizes=np.stack(prefix_sizes,1),teacher_ce=np.stack(teacher_ce,1),lengths=lengths.numpy(),
                  teacher_full_loss=(teacher_tokens.sum(1)/(lengths-1)).numpy())
    if return_embeddings:
        result.update(original_trajectory=np.stack(original_rows,axis=1),
                      generated_trajectories=np.stack(generated_rows,axis=2))
    return result,greedy_texts


def sustained_threshold(probabilities,cutoff=.5):
    """Smallest tested prefix with success >=cutoff at this AND all later points."""
    sustained=np.minimum.accumulate(probabilities[...,::-1],axis=-1)[...,::-1]>=cutoff
    index=sustained.argmax(-1)
    return np.where(sustained.any(-1),FRACTIONS[index],np.inf)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/membrain_recovery'))
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--samples',type=int,default=16);args=p.parse_args()
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    texts,background=sentence_pool();x,_=encode(texts);bx,_=encode(background)
    rng=np.random.default_rng(90121);groups=rng.permutation(np.repeat(np.arange(4),32))
    (args.output/'sentences.json').write_text(json.dumps(dict(evaluation=texts,background=background),indent=2))
    for rotation in range(4):
        seed=51001+rotation;path=args.output/f'model_{seed}.npz'
        cached_values=None
        if path.exists():
            with np.load(path) as cached:
                assert int(cached['epochs'])==args.epochs and int(cached['samples'])==args.samples
                if cached['prefix_sizes'].shape[1]==len(FRACTIONS):
                    print(f'Reusing model {seed}',flush=True);continue
                cached_values={k:cached[k] for k in cached.files}
        torch.manual_seed(seed);lm=SentenceLM()
        repeats=LEVELS[(groups+rotation)%4]
        if cached_values is None:
            print(f'Membrain recovery model {rotation+1}/4: background language training',flush=True)
            background_history=train(lm,bx,np.ones(len(bx),dtype=int),10)
            print(f'Membrain recovery model {rotation+1}/4: controlled sentence exposure',flush=True)
            history=train(lm,x,repeats,args.epochs)
            torch.save(lm.state_dict(),args.output/f'model_{seed}.pt')
        else:
            print(f'Recomputing recovery grid from saved weights for {seed}',flush=True)
            lm.load_state_dict(torch.load(args.output/f'model_{seed}.pt',weights_only=True))
            history=cached_values['history'];background_history=cached_values['background_history']
        torch.manual_seed(seed+10000)
        values,generated=recover(lm,texts,args.samples)
        values.update(repeats=repeats,presentations=repeats*args.epochs,fractions=FRACTIONS,
                      history=np.array(history),background_history=np.array(background_history),
                      epochs=np.array(args.epochs),samples=np.array(args.samples),seed=np.array(seed))
        np.savez_compressed(path,**values)
        (args.output/f'generations_{seed}.json').write_text(json.dumps(dict(seed=seed,prefix_fractions=FRACTIONS.tolist(),
            greedy_suffixes=generated),indent=2))
        print(f'Completed model {seed}',flush=True)
    (args.output/'config.json').write_text(json.dumps(dict(epochs=args.epochs,samples=args.samples,
        repeats_per_epoch=LEVELS.tolist(),total_presentations=(LEVELS*args.epochs).tolist(),
        fractions=FRACTIONS.tolist(),tail_chars=TAIL,background_sentences=128,background_epochs=10,
        evaluation_sentences=128,length_range_chars=[80,120],seeds=list(range(51001,51005)),
        design='four-model balanced exposure rotation; every sentence at every level once',
        source_sha256=previous.SOURCE_SHA,torch=torch.__version__,numpy=np.__version__),indent=2))
    print('DONE',flush=True)


if __name__=='__main__':main()
