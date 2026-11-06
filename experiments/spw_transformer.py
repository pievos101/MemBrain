"""Small causal Transformer and batched probability oracle for CPU benchmarks."""
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__(); self.heads=heads
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d)
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))

    def forward(self,x,cache=None,causal=True):
        n,t,d=x.shape
        q,k,v=self.qkv(self.ln1(x)).reshape(n,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        if cache is not None:
            k=torch.cat([cache[0],k],dim=2);v=torch.cat([cache[1],v],dim=2)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=causal and cache is None)
        x=x+self.proj(a.transpose(1,2).reshape(n,t,d));x=x+self.ff(self.ln2(x))
        return x,(k,v)


class TinyTransformer(nn.Module):
    def __init__(self,vocab=1024,d=64,layers=2,heads=4,max_length=256):
        super().__init__();self.vocab=vocab;self.d=d;self.max_length=max_length
        self.embedding=nn.Embedding(vocab,d);self.position=nn.Embedding(max_length,d)
        self.blocks=nn.ModuleList([Block(d,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(d);self.output=nn.Linear(d,vocab,bias=False)
        self.output.weight=self.embedding.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Embedding,nn.Linear)):
            nn.init.normal_(m.weight,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None:nn.init.zeros_(m.bias)

    def forward(self,ids,cache=None,causal=True,return_hidden=False):
        if cache is not None and ids.shape[1]!=1:
            raise ValueError('Cached decoding accepts one new token at a time.')
        offset=0 if cache is None else cache[0][0].shape[2]
        if offset+ids.shape[1]>self.max_length:raise ValueError('Context too long')
        x=self.embedding(ids)+self.position(torch.arange(offset,offset+ids.shape[1],device=ids.device))
        new=[]
        for i,b in enumerate(self.blocks):
            x,c=b(x,None if cache is None else cache[i],causal);new.append(c)
        h=self.norm(x);logits=self.output(h)
        # PAD/UNK/MASK are never legal text outputs, including during training.
        logits=logits.clone();logits[:,:,:3]=-10000.
        if return_hidden:return logits,h,new
        return logits


class Oracle:
    def __init__(self,model,token_strings,cache_limit=16000,batch_size=128):
        self.model=model.eval();self.token_strings=token_strings
        self.cache=OrderedDict();self.limit=cache_limit;self.batch=batch_size
        self.stats=dict(context_requests=0,unique_context_evaluations=0,forward_batches=0)

    @torch.no_grad()
    def teacher(self,sequences):
        x=torch.as_tensor(np.asarray(sequences),dtype=torch.long)
        out=[]
        for b in x.split(self.batch):
            out.append(self.model(b[:,:-1]).log_softmax(-1).numpy())
            self.stats['forward_batches']+=1
        return np.concatenate(out)

    @torch.no_grad()
    def distributions(self,contexts):
        self.stats['context_requests']+=len(contexts)
        missing=list(dict.fromkeys(c for c in contexts if c not in self.cache));fresh={}
        by_length={}
        for c in missing:by_length.setdefault(len(c),[]).append(c)
        for group in by_length.values():
            for start in range(0,len(group),self.batch):
                keys=group[start:start+self.batch]
                lp=self.model(torch.tensor(keys,dtype=torch.long))[:,-1].log_softmax(-1).numpy()
                self.stats['forward_batches']+=1
                self.stats['unique_context_evaluations']+=len(keys)
                for c,p in zip(keys,lp):fresh[c]=p
        result=[fresh[c] if c in fresh else self.cache[c] for c in contexts]
        self.cache.update(fresh)
        while len(self.cache)>self.limit:self.cache.popitem(last=False)
        return result

    @torch.no_grad()
    def nll(self,sequences):
        result=np.zeros(len(sequences));groups={}
        for i,x in enumerate(sequences):
            if len(x)<2:raise ValueError('Need two tokens')
            groups.setdefault(len(x),[]).append(i)
        for ids in groups.values():
            for start in range(0,len(ids),self.batch):
                subset=ids[start:start+self.batch];x=np.asarray([sequences[i] for i in subset])
                lp=self.teacher(x)
                result[subset]=-np.take_along_axis(lp,x[:,1:,None],axis=2).mean((1,2))
        return result


def train_lm(model,x,epochs,seed,lr=.003,mlm=False,batch_size=32):
    generator=torch.Generator().manual_seed(seed);x=torch.as_tensor(np.asarray(x),dtype=torch.long)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01);history=[]
    model.train()
    for epoch in range(epochs):
        total=0.;tokens=0
        for idx in torch.randperm(len(x),generator=generator).split(batch_size):
            b=x[idx]
            if mlm:
                mask=torch.rand(b.shape,generator=generator)<.15
                inp=b.clone();inp[mask]=2
                logits=model(inp,causal=False)
                loss=F.cross_entropy(logits[mask],b[mask]);count=int(mask.sum())
            else:
                logits=model(b[:,:-1]);loss=F.cross_entropy(logits.flatten(0,1),b[:,1:].flatten());count=b[:,1:].numel()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=loss.item()*count;tokens+=count
        history.append(total/tokens)
        print(f'    epoch {epoch+1}/{epochs}: {history[-1]:.4f}',flush=True)
    model.eval();return history


@torch.no_grad()
def trajectories(model,x,fractions):
    """Mean final hidden states AFTER each true/generated whole-suffix token."""
    x=torch.as_tensor(np.asarray(x),dtype=torch.long);n,length=x.shape
    true=[];generated=[];accuracy=[];exact=[];rollout=[]
    for f in fractions:
        p=max(1,min(length-1,round(length*f)));tr=[];ge=[];ac=[];ex=[];ce=[]
        for b in x.split(64):
            _,h,_=model(b,return_hidden=True);tr.append(h[:,p:].mean(1).numpy())
            logits,_,cache=model(b[:,:p],return_hidden=True)
            hs=[];ts=[];losses=[]
            for j in range(p,length):
                losses.append(F.cross_entropy(logits[:,-1],b[:,j],reduction='none'))
                token=logits[:,-1].argmax(-1);ts.append(token)
                logits,h,cache=model(token[:,None],cache=cache,return_hidden=True);hs.append(h[:,0])
            match=torch.stack(ts,1)==b[:,p:]
            ge.append(torch.stack(hs,1).mean(1).numpy());ac.append(match.float().mean(1).numpy())
            ex.append(match.all(1).float().numpy());ce.append(torch.stack(losses,1).mean(1).numpy())
        true.append(np.concatenate(tr));generated.append(np.concatenate(ge));accuracy.append(np.concatenate(ac))
        exact.append(np.concatenate(ex));rollout.append(np.concatenate(ce))
    return dict(true=np.stack(true,1),generated=np.stack(generated,1),accuracy=np.stack(accuracy,1),
                exact=np.stack(exact,1),rollout_ce=np.stack(rollout,1))
