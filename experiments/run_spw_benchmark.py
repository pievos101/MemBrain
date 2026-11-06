"""New, small-scale natural-text benchmarks; NOT original-paper reproduction.

See SPW_BENCHMARK.md for the predeclared protocol and implementation choices.
Run from the repository root using the project Python environment.
"""
import argparse
import copy
import csv
import hashlib
import json
import re
import platform
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import tokenizers
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from spw_transformer import TinyTransformer,Oracle,train_lm,trajectories
from spw_methods import (spw_batch,loss_score,min_k_score,zlib_score,lowercase_score,
                         neighbor_score,evaluate_scores,component_ablations)

FRACTIONS=[.05,.1,.2,.3,.4,.5,.6,.7,.8,.9]
SOURCES={
 'ag_news':dict(file='ag_news_train.csv',sha256='76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e',
 url='https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv'),
 'wikitext2':dict(file='wikitext2_train.txt',sha256='9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f',
 url='https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt')}


def save_json(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def prepare(root,length=48):
    pools={};background={};manifest={'sources':SOURCES,'length':length,'splits':{}}
    if all((root/f).exists() for f in ['tokenizer.json','benchmark_data.npz','data_manifest.json']):
        saved=json.loads((root/'data_manifest.json').read_text())
        if saved['sources']!=SOURCES or saved['length']!=length:raise ValueError('Use a fresh output for a different protocol')
        with np.load(root/'benchmark_data.npz') as arrays:
            bg=arrays['background'].copy();datasets={k:arrays[k].copy() for k in SOURCES}
        for name,x in datasets.items():
            if hashlib.sha256(x.tobytes()).hexdigest()!=saved['splits'][name]['pool_sha256']:
                raise ValueError('Saved split checksum mismatch')
        return Tokenizer.from_file(str(root/'tokenizer.json')),bg,datasets
    for name,spec in SOURCES.items():
        path=root/'data'/spec['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
        if name=='ag_news':
            with path.open() as f:texts=[r[2] for r in csv.reader(f)]
        else:texts=[s for s in path.read_text().splitlines() if not s.strip().startswith('=')]
        texts=list(dict.fromkeys(re.sub(r'\s+',' ',s).strip() for s in texts))
        texts=[s for s in texts if len(s)>=300]
        order=np.random.default_rng(8341).permutation(len(texts));texts=[texts[i] for i in order]
        background[name]=texts[:2048];pools[name]=texts[2048:]
    tok=Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder=decoders.ByteLevel()
    tok.train_from_iterator([s for pool in background.values() for s in pool],
       trainers.BpeTrainer(vocab_size=1024,special_tokens=['[PAD]','[UNK]','[MASK]'],
                           initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(root/'tokenizer.json'))
    seen=set();bg=[]
    # Exclude every tokenizer-background chunk from ALL subsequent roles.
    for name,pool in background.items():
        encoded=[tok.encode(s).ids[:length] for s in pool]
        seen.update(tuple(x) for x in encoded)
        bg.extend([x for x in encoded if len(x)==length][:512])
    datasets={}
    for name,pool in pools.items():
        chunks=[]
        for s in pool:
            x=tok.encode(s).ids[:length];key=tuple(x)
            if len(x)!=length or key in seen:continue
            seen.add(key);chunks.append(x)
            if len(chunks)==576:break
        assert len(chunks)==576
        x=np.array(chunks,dtype=np.int64);datasets[name]=x
        manifest['splits'][name]=dict(pool_sha256=hashlib.sha256(x.tobytes()).hexdigest(),count=len(x))
    np.savez_compressed(root/'benchmark_data.npz',background=np.array(bg),**datasets)
    save_json(root/'data_manifest.json',manifest)
    return tok,np.array(bg),datasets


@torch.no_grad()
def make_neighbors(model,x,seed):
    """10 unique single-token replacements from a disjoint-background masked LM."""
    rng=np.random.default_rng(seed);out=[]
    for b in np.array_split(x,max(1,(len(x)+31)//32)):
        positions=np.stack([rng.choice(x.shape[1],10,replace=False) for _ in b])
        repeated=np.repeat(b,10,axis=0);pos=positions.reshape(-1)
        masked=torch.tensor(repeated.copy());masked[torch.arange(len(masked)),torch.tensor(pos)]=2
        choices=[]
        for start in range(0,len(masked),64):
            logits=model(masked[start:start+64],causal=False)
            row=torch.arange(len(logits));p=torch.tensor(pos[start:start+64])
            pred=logits[row,p].clone()
            original=torch.tensor(repeated[start:start+64])[row,p];pred[row,original]=-10000.
            choices.append(pred.argmax(1).numpy())
        repeated[np.arange(len(repeated)),pos]=np.concatenate(choices)
        out.append(repeated.reshape(len(b),10,x.shape[1]))
    return np.concatenate(out)


class OriginCNN(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(d,32,3,padding=1),nn.GELU(),nn.Conv1d(32,16,3,padding=1),
                               nn.GELU(),nn.Flatten(),nn.Linear(160,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x.transpose(1,2))[:,0]


def origin_classifier(t,seed):
    """Auxiliary origins only: 96 train,32 early stop,32 calibration,32 held out."""
    torch.manual_seed(seed);a=t['true'];b=t['generated']
    mu=torch.tensor(np.concatenate([a[:96],b[:96]]).mean((0,1)),dtype=torch.float32)
    sd=torch.tensor(np.concatenate([a[:96],b[:96]]).std((0,1)),dtype=torch.float32).clamp_min(.1)
    xa=(torch.tensor(a)-mu)/sd;xb=(torch.tensor(b)-mu)/sd
    def pairs(lo,hi):
        return torch.cat([xa[lo:hi],xb[lo:hi]]),torch.cat([torch.zeros(hi-lo),torch.ones(hi-lo)])
    train,y=pairs(0,96);valid,vy=pairs(96,128)
    model=OriginCNN(a.shape[-1]);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best=None;lowest=float('inf');best_epoch=0
    for epoch in range(80):
        model.train()
        for ids in torch.randperm(len(train)).split(32):
            loss=F.binary_cross_entropy_with_logits(model(train[ids]),y[ids]);opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():v=F.binary_cross_entropy_with_logits(model(valid),vy).item()
        if v<lowest:lowest=v;best=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        if epoch+1-best_epoch>=12:break
    model.load_state_dict(best);model.eval()
    # Scalar temperature chosen using only the separate origin calibration set.
    cx,cy=pairs(128,160)
    with torch.no_grad():cl=model(cx)
    temperatures=np.geomspace(.25,8,50)
    temperature=float(min(temperatures,key=lambda z:F.binary_cross_entropy_with_logits(cl/z,cy).item()))
    with torch.no_grad():
        hx,hy=pairs(160,192);hp=(model(hx)/temperature).sigmoid().numpy()
    diagnostic=evaluate_scores(hy.numpy(),hp)
    diagnostic.update(best_epoch=best_epoch,temperature=temperature)
    def predict(traj):
        with torch.no_grad():return (model((torch.tensor(traj)-mu)/sd)/temperature).sigmoid().numpy()
    return predict,diagnostic,dict(state=best,mu=mu,sd=sd,temperature=temperature)


def bootstrap(y,scores,seed,count=1000):
    rng=np.random.default_rng(seed);pos=np.flatnonzero(y);neg=np.flatnonzero(1-y)
    # Pair every draw across methods, preserving the class counts.
    draws=[np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))] for _ in range(count)]
    # Rank-based AUC avoids repeatedly constructing full ROC/PR curves.
    def auc(labels,values):
        order=np.argsort(values,kind='stable');s=values[order];lab=labels[order]
        starts=np.r_[0,np.flatnonzero(np.diff(s))+1];ends=np.r_[starts[1:],len(s)]
        ranks=np.repeat((starts+ends+1)/2,ends-starts)
        n=labels.sum();return (ranks[lab==1].sum()-n*(n+1)/2)/(n*(len(labels)-n))
    values={m:np.array([auc(y[d],s[d]) for d in draws]) for m,s in scores.items()}
    return {m:dict(auc_ci95=np.quantile(v,[.025,.975]).tolist(),
         auc_difference_from_loss_ci95=np.quantile(v-values['LOSS'],[.025,.975]).tolist()) for m,v in values.items()}


def run_cell(root,tok,base,x,neighbors,corpus,seed,regime,epochs):
    name=f'{corpus}_seed{seed}_{regime}';path=root/name;path.mkdir(exist_ok=True)
    if (path/'metrics.json').exists():
        print('SKIP completed',name,flush=True);return
    started=time.perf_counter();rng=np.random.default_rng(seed)
    # Fixed pool, fresh random membership assignment for each independent seed.
    order=rng.permutation(384);members=order[:256];nonmembers=order[256:]
    eval_ids=np.r_[members[:128],nonmembers];y=np.r_[np.ones(128,dtype=int),np.zeros(128,dtype=int)]
    test=x[eval_ids];aux=x[384:576]
    model=copy.deepcopy(base)
    train=x[members]
    if regime=='repeated':train=np.concatenate([train,np.repeat(train[:32],7,axis=0)])
    history=train_lm(model,train,epochs,seed+71)
    torch.save(model.state_dict(),path/'target.pt')
    words=[tok.decode([i],skip_special_tokens=False) for i in range(tok.get_vocab_size())]
    oracle=Oracle(model,words);lp=oracle.teacher(test)
    token_lp=np.take_along_axis(lp,test[:,1:,None],axis=2)[:,:,0];nll=-token_lp.mean(1)
    texts=[tok.decode(row.tolist()) for row in test]
    lower=[tok.encode(s.lower()).ids for s in texts];lower_nll=oracle.nll(lower)
    neigh_nll=oracle.nll(neighbors[eval_ids].reshape(-1,test.shape[1])).reshape(len(test),10)
    print('  SPW adaptive shortening',name,flush=True);spw_start=time.perf_counter()
    details=spw_batch(oracle,test,teacher_log_probs=lp);spw_seconds=time.perf_counter()-spw_start
    del lp
    print('  Membrain whole-suffix trajectories',name,flush=True);mem_start=time.perf_counter()
    aux_t=trajectories(model,aux,FRACTIONS);test_t=trajectories(model,test,FRACTIONS)
    predict,origin_diag,classifier=origin_classifier(aux_t,seed+923)
    torch.save(classifier,path/'origin_classifier.pt')
    qt,qg=predict(test_t['true']),predict(test_t['generated'])
    # Declared primary: convergence of origin probabilities, never tuned on membership.
    scores={'LOSS':np.array([loss_score(p) for p in token_lp]),
      'Min-k20%':np.array([min_k_score(p) for p in token_lp]),
      'Zlib':np.array([zlib_score(a,b) for a,b in zip(nll,texts)]),
      'Lowercase':np.array([lowercase_score(a,b) for a,b in zip(nll,lower_nll)]),
      'Neighbor':np.array([neighbor_score(a,b) for a,b in zip(nll,neigh_nll)]),
      'SPW-MIA':np.array([c['spw'] for c in details]),'Membrain':-np.abs(qt-qg)}
    secondary={'Membrain true-origin score':qt,'Membrain signed convergence':qt-qg,
               'Membrain cosine':np.mean(np.sum(test_t['true']*test_t['generated'],2)/
                 (np.linalg.norm(test_t['true'],axis=2)*np.linalg.norm(test_t['generated'],axis=2)),1)}
    mem_seconds=time.perf_counter()-mem_start
    np.savez_compressed(path/'scores.npz',labels=y,eval_ids=eval_ids,**scores,**secondary)
    np.savez_compressed(path/'trajectories.npz',labels=y,**test_t,qtrue=qt,qgenerated=qg)
    boot=bootstrap(y,scores,seed)
    metrics={m:dict(**evaluate_scores(y,s),**boot[m]) for m,s in scores.items()}
    ablations={m:evaluate_scores(y,np.array([component_ablations(c)[m] for c in details]))
               for m in component_ablations(details[0])}
    imbalances={};pos=np.flatnonzero(y);neg=np.flatnonzero(1-y)
    for gamma in [.05,.1,.2,.4,.8,1.]:
        draws=[np.r_[rng.choice(pos,max(1,round(gamma*len(neg))),replace=False),neg] for _ in range(20)]
        imbalances[str(gamma)]={m:dict(mean_ap=float(np.mean([evaluate_scores(y[d],s[d])['average_precision'] for d in draws])),
                 sd_ap=float(np.std([evaluate_scores(y[d],s[d])['average_precision'] for d in draws]))) for m,s in scores.items()}
    exposures=np.r_[np.full(32,epochs*(8 if regime=='repeated' else 1)),np.full(96,epochs),np.zeros(128)]
    np.save(path/'exposures.npy',exposures)
    result=dict(corpus=corpus,seed=seed,regime=regime,history=history,metrics=metrics,
      secondary_metrics={m:evaluate_scores(y,s) for m,s in secondary.items()},component_ablations=ablations,
      imbalances=imbalances,origin_holdout=origin_diag,oracle_stats=oracle.stats,
      spw_seconds=spw_seconds,membrain_seconds=mem_seconds,total_seconds=time.perf_counter()-started,
      mean_shortened_ratio=float(np.mean([c['mean_shortened_ratio'] for c in details])),
      parameter_count=sum(p.numel() for p in model.parameters()),
      membership_split_sha256=hashlib.sha256(order.tobytes()).hexdigest())
    save_json(path/'spw_details.json',details);save_json(path/'metrics.json',result)
    print(name,{m:round(v['auc'],3) for m,v in metrics.items()},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/spw_scale'))
    p.add_argument('--seeds',type=int,nargs='+',default=[101,202]);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--download-data',action='store_true',help='Download public data with checksum verification')
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    if args.download_data:
        (root/'data').mkdir(exist_ok=True)
        for spec in SOURCES.values():
            path=root/'data'/spec['file']
            if not path.exists():
                with urllib.request.urlopen(spec['url'],timeout=120) as response:
                    payload=response.read()
                if hashlib.sha256(payload).hexdigest()!=spec['sha256']:raise ValueError('Source checksum mismatch')
                path.write_bytes(payload)
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    tok,bg,datasets=prepare(root)
    config=dict(description='Own CPU benchmark; not original-paper numerical reproduction',seeds=args.seeds,
      architecture=dict(type='causal Transformer',dimension=64,layers=2,heads=4,vocab=1024,context=48),
      pretraining_epochs=3,standard_epochs=4,repeated_epochs=12,additional_repeat_factor=8,
      target_members=256,evaluation_members=128,evaluation_nonmembers=128,auxiliary=192,
      fractions=FRACTIONS,bootstrap_draws=1000,threads=args.threads)
    config['software']=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,tokenizers=tokenizers.__version__)
    save_json(root/'config.json',config)
    torch.manual_seed(7551);proposal=TinyTransformer()
    print('Training shared public-background masked proposal model',flush=True)
    proposal_path=root/'proposal.pt'
    if proposal_path.exists():proposal.load_state_dict(torch.load(proposal_path,weights_only=True))
    else:train_lm(proposal,bg,4,7551,mlm=True);torch.save(proposal.state_dict(),proposal_path)
    proposal.eval();neighbors={}
    for name,x in datasets.items():
        path=root/f'{name}_neighbors.npy'
        if path.exists():neighbors[name]=np.load(path)
        else:neighbors[name]=make_neighbors(proposal,x[:384],2871);np.save(path,neighbors[name])
    for seed in args.seeds:
        torch.manual_seed(seed);base=TinyTransformer();path=root/f'base_seed{seed}.pt'
        if path.exists():base.load_state_dict(torch.load(path,weights_only=True))
        else:
            print('Public background causal pretraining seed',seed,flush=True)
            train_lm(base,bg,3,seed);torch.save(base.state_dict(),path)
        for name,x in datasets.items():
            for regime,epochs in [('standard',4),('repeated',12)]:
                print('CELL',name,seed,regime,flush=True)
                run_cell(root,tok,base,x,neighbors[name],name,seed,regime,epochs)


if __name__=='__main__':main()
