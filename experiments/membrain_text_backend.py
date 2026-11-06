"""Local stand-in for an opaque text-completion service. Internals stay HERE.

The attacker receives only strings. Tokenizer, weights and sampling probabilities
are backend implementation details, just as in a hosted text-only API.
"""
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F


class TextCompletionService:
    def __init__(self,model,tokenizer):
        self.model=model.eval();self.tokenizer=tokenizer
        eos=tokenizer.encode('\n').ids
        if len(eos)!=1:raise ValueError('Need single newline terminator token')
        self.eos=eos[0];self.stats=dict(prefix_requests=0,completions=0,generated_tokens=0)

    @torch.no_grad()
    def __call__(self,prefixes,samples,temperature,max_new_tokens,seed):
        rng=torch.Generator().manual_seed(seed);groups=defaultdict(list);paths=samples+1
        result=[dict(greedy='',samples=['']*samples) for _ in prefixes]
        for i,prefix in enumerate(prefixes):
            ids=self.tokenizer.encode(prefix).ids
            if not ids or len(ids)+max_new_tokens>self.model.max_length:raise ValueError('Invalid prompt/context budget')
            for path in range(paths):groups[len(ids)].append((i,path,ids))
        for group in groups.values():
            for start in range(0,len(group),96):
                entries=group[start:start+96];x=torch.tensor([a[2] for a in entries]);tokens=[[] for _ in entries]
                logits,_,cache=self.model(x,return_hidden=True);active=np.arange(len(entries))
                for _ in range(max_new_tokens):
                    distribution=logits[:,-1]/temperature
                    tok=torch.multinomial(distribution.softmax(-1),1,generator=rng)[:,0]
                    greedy=torch.tensor([entries[i][1]==0 for i in active])
                    tok[greedy]=logits[:,-1].argmax(-1)[greedy]
                    self.stats['generated_tokens']+=len(tok)
                    continuing=tok!=self.eos
                    for index,value in zip(active,tok.tolist()):
                        if value!=self.eos:tokens[index].append(value)
                    if not continuing.any():break
                    active=active[continuing.numpy()]
                    cache=[(k[continuing],v[continuing]) for k,v in cache]
                    logits,_,cache=self.model(tok[continuing,None],cache=cache,return_hidden=True)
                for (index,path,_),out in zip(entries,tokens):
                    text=self.tokenizer.decode(out)
                    if path==0:result[index]['greedy']=text
                    else:result[index]['samples'][path-1]=text
        self.stats['prefix_requests']+=len(prefixes);self.stats['completions']+=len(prefixes)*paths
        return result


def train_sentences(model,sequences,epochs,seed):
    """Benchmark-owner target training; PAD targets ignored, no attack fitting."""
    rng=torch.Generator().manual_seed(seed);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    model.train();history=[]
    for epoch in range(epochs):
        total=0.;count=0
        for ids in torch.randperm(len(sequences),generator=rng).split(32):
            rows=[sequences[i] for i in ids];length=max(map(len,rows));x=torch.zeros(len(rows),length,dtype=torch.long)
            for i,row in enumerate(rows):x[i,:len(row)]=torch.tensor(row)
            pred=model(x[:,:-1]);target=x[:,1:];loss=F.cross_entropy(pred.flatten(0,1),target.flatten(),ignore_index=0)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            n=int((target!=0).sum());total+=loss.item()*n;count+=n
        history.append(total/count);print(f'  sentence epoch{epoch+1}/{epochs}: {history[-1]:.4f}',flush=True)
    model.eval();return history
