"""Controlled synthetic membership experiment; no membership labels train attacks.

Run with Python 3.12, numpy 2.2.6, torch 2.6.0+cpu, matplotlib 3.10.8:
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python experiments/corrected_benchmark.py
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PREFIXES = [8, 16, 24, 32]
HORIZON = 8
VOCAB = 53  # Fixed, public vocabulary: never fitted to any split.


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 24)
        self.gru = nn.GRU(24, 48, batch_first=True)
        self.output = nn.Linear(48, VOCAB)

    def forward(self, x):
        h, _ = self.gru(self.embedding(x))
        return self.output(h)


class Attack(nn.Module):
    def __init__(self, ordered):
        super().__init__()
        self.ordered = ordered
        self.layers = (nn.Sequential(nn.Conv1d(48, 24, 3, padding=1), nn.ReLU(),
                                    nn.Conv1d(24, 24, 3, padding=1), nn.ReLU())
                       if ordered else nn.Sequential(nn.Linear(48, 24), nn.ReLU(),
                                                    nn.Linear(24, 24), nn.ReLU()))
        self.head = nn.Linear(24, 2)

    def forward(self, x):
        h = self.layers(x.transpose(1, 2)).mean(2) if self.ordered else self.layers(x).mean(1)
        return self.head(h)


def corpus(seed):
    """One common Markov source, deduplicated before random split assignment."""
    rng = np.random.default_rng(seed)
    # Mostly predictable shared grammar, with stochastic transitions that give
    # individual documents distinct realizations. Both classes use this source.
    transition = rng.dirichlet(np.ones(VOCAB) * .12, size=VOCAB)
    docs, seen = [], set()
    while len(docs) < 340:
        x = [int(rng.integers(VOCAB))]
        for _ in range(47):
            x.append(int(rng.choice(VOCAB, p=transition[x[-1]])))
        key = tuple(x)
        if key not in seen:
            seen.add(key)
            docs.append(x)
    x = torch.tensor(docs)[torch.tensor(rng.permutation(len(docs)))]
    groups = {'ordinary': x[:80], 'strong': x[80:100], 'nonmember': x[100:200],
              'aux_train': x[200:300], 'aux_val': x[300:340]}
    sets = {k: {tuple(row.tolist()) for row in v} for k, v in groups.items()}
    assert all(len(sets[k]) == len(v) for k, v in groups.items())
    keys = list(sets)
    assert all(not sets[a] & sets[b] for i, a in enumerate(keys) for b in keys[i+1:])
    return x, groups


def train_lm(model, groups, epochs):
    # Equal ordinary exposure; strong documents deliberately appear 15x.
    data = torch.cat([groups['ordinary'], groups['strong'].repeat(15, 1)])
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    model.train()
    history = []
    for epoch in range(epochs):
        values = []
        for ids in torch.randperm(len(data)).split(32):
            batch = data[ids]
            loss = F.cross_entropy(model(batch[:, :-1]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); values.append(loss.item())
        history.append(float(np.mean(values)))
    return history


@torch.no_grad()
def features(model, docs, samples):
    model.eval()
    observed, generated = [], []
    for p in PREFIXES:
        # State after consuming each of the next HORIZON tokens; equal horizon,
        # causal context and pooling for observed and sampled continuations.
        h, _ = model.gru(model.embedding(docs[:, :p+HORIZON]))
        observed.append(h[:, p:p+HORIZON].mean(1))
        prefix = docs[:, :p].repeat_interleave(samples, dim=0)
        _, state = model.gru(model.embedding(prefix))
        hs = []
        for _ in range(HORIZON):
            probs = (model.output(state[-1]) / 1.15).softmax(-1)
            token = torch.multinomial(probs, 1)
            h, state = model.gru(model.embedding(token), state)
            hs.append(h[:, 0])
        generated.append(torch.stack(hs, 1).mean(1).reshape(len(docs), samples, 48))
    observed = torch.stack(observed, 1)
    generated = torch.stack(generated, 2)  # document, sample, prefix, hidden
    loss = F.cross_entropy(model(docs[:, :-1]).transpose(1, 2), docs[:, 1:], reduction='none').mean(1)
    return observed, generated, -loss.numpy()


def train_attack(original, generated, ordered):
    # Only auxiliary, nonmember documents train/validate original-vs-generated.
    # One sampled trajectory per document avoids treating dependent samples as
    # extra independent documents. Evaluation averages all sampled trajectories.
    train = torch.cat([original[200:300], generated[200:300, 0]])
    val = torch.cat([original[300:], generated[300:, 0]])
    y = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    vy = torch.cat([torch.zeros(40), torch.ones(40)]).long()
    mean = train.mean((0, 1), keepdim=True)
    sd = train.std((0, 1), keepdim=True).clamp_min(.01)
    train, val = (train-mean)/sd, (val-mean)/sd
    model = Attack(ordered)
    opt = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=.001)
    best, saved, stale = float('inf'), None, 0
    for _ in range(100):
        model.train()
        for ids in torch.randperm(len(train)).split(32):
            loss = F.cross_entropy(model(train[ids]), y[ids])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            value = F.cross_entropy(model(val), vy).item()
        if value < best - 1e-5:
            best, saved, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 12:
            break
    model.load_state_dict(saved); model.eval()
    with torch.no_grad():
        accuracy = (model(val).argmax(1) == vy).float().mean().item()
        po = model((original-mean)/sd).softmax(1)[:, 0]
        g = ((generated-mean.unsqueeze(1))/sd.unsqueeze(1)).reshape(-1, 4, 48)
        pg = model(g).softmax(1)[:, 0].reshape(len(original), -1).mean(1)
    return (po-pg).numpy(), accuracy


def auc(positive, negative):
    d = positive[:, None] - negative[None, :]
    return float(((d > 0) + .5*(d == 0)).mean())


def seed_run(seed, args):
    torch.manual_seed(seed)
    docs, groups = corpus(seed)
    model = LM()
    initial = copy.deepcopy(model)
    history = train_lm(model, groups, args.epochs)
    scores = {}
    diagnostics = {}
    for condition, lm in [('untrained', initial), ('trained', model)]:
        torch.manual_seed(seed+10000)  # Separate generation RNG from training.
        observed, generated, loss = features(lm, docs, args.samples)
        mu = generated.mean(1)
        variance = generated.var(1, unbiased=True)
        # Per-prefix variance floor estimated using auxiliary training data only.
        floor = .1*variance[200:300].mean((0, 2), keepdim=True)
        distance = ((observed-mu).square()/(variance+floor.clamp_min(1e-6))).mean(2).numpy()
        affinity = -distance  # Predefined: more typical continuation -> higher membership.
        cnn, cnn_acc = train_attack(observed, generated, True)
        pooled, pool_acc = train_attack(observed, generated, False)
        methods = {'LM loss': loss, 'Mean affinity': affinity.mean(1),
                   'Late affinity (p32)': affinity[:, -1],
                   'Affinity slope': (affinity*(np.arange(4)-1.5)).sum(1)/5,
                   'CNN original gap': cnn, 'Pooled original gap': pooled}
        for j, p in enumerate(PREFIXES):
            methods[f'Affinity p{p}'] = affinity[:, j]
        scores[condition] = {k: v[:200].tolist() for k, v in methods.items()}
        diagnostics[condition] = {'cnn_aux_accuracy': cnn_acc, 'pooled_aux_accuracy': pool_acc}
    return {'seed': seed, 'scores': scores, 'diagnostics': diagnostics, 'lm_loss_history': history,
            'split_sizes': {k: len(v) for k, v in groups.items()}, 'overlapping_documents': 0}


def summarize(runs, out):
    rng = np.random.default_rng(72581)
    rows = []
    for condition in ['untrained', 'trained']:
        for group, ids in [('ordinary', np.arange(80)), ('strong', np.arange(80, 100))]:
            for method in runs[0]['scores'][condition]:
                arrays = [np.array(r['scores'][condition][method]) for r in runs]
                values = [auc(a[ids], a[100:200]) for a in arrays]
                boot = []
                # Hierarchical bootstrap: seed, then documents within each seed.
                for _ in range(1000):
                    boot.append(np.mean([auc(arrays[i][rng.choice(ids, len(ids))],
                                             arrays[i][rng.integers(100, 200, 100)])
                                         for i in rng.integers(len(runs), size=len(runs))]))
                low, high = np.quantile(boot, [.025, .975])
                rows.append(dict(condition=condition, group=group, method=method,
                                 auc=np.mean(values), ci_low=low, ci_high=high,
                                 seed_sd=np.std(values, ddof=1)))
    with (out/'auc_summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # Paired comparison with the same bootstrap observations for both methods.
    contrasts = []
    for group, ids in [('ordinary', np.arange(80)), ('strong', np.arange(80,100))]:
        for left, right in [('Mean affinity','Late affinity (p32)'),
                            ('Affinity slope','Mean affinity'),
                            ('CNN original gap','Pooled original gap'),
                            ('Mean affinity','LM loss')]:
            def difference(i, p, n):
                a = np.array(runs[i]['scores']['trained'][left])
                b = np.array(runs[i]['scores']['trained'][right])
                return auc(a[p], a[n])-auc(b[p], b[n])
            point = np.mean([difference(i, ids, np.arange(100,200)) for i in range(len(runs))])
            boot = [np.mean([difference(i, rng.choice(ids,len(ids)),rng.integers(100,200,100))
                             for i in rng.integers(len(runs),size=len(runs))]) for _ in range(1000)]
            lo, hi = np.quantile(boot,[.025,.975])
            contrasts.append(dict(group=group, comparison=f'{left} minus {right}', delta_auc=point, ci_low=lo, ci_high=hi))
    with (out/'paired_comparisons.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(contrasts[0])); w.writeheader(); w.writerows(contrasts)
    return rows


def plots(runs, rows, out):
    os.environ.setdefault('MPLCONFIGDIR', '/workspace/.cache/matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME', '/workspace/.cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':180})
    methods = ['LM loss','Mean affinity','Late affinity (p32)','Affinity slope','CNN original gap','Pooled original gap']
    def save(fig, name):
        fig.savefig(out/f'{name}.png',bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax, group in zip(axes,['ordinary','strong']):
        for c, offset, color in [('untrained',-.12,'#999999'),('trained',.12,'#0072B2')]:
            r = [next(x for x in rows if x['condition']==c and x['group']==group and x['method']==m) for m in methods]
            y=np.array([x['auc'] for x in r]); lo=np.array([x['ci_low'] for x in r]); hi=np.array([x['ci_high'] for x in r])
            ax.errorbar(y,np.arange(len(methods))+offset,xerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt='o',color=color,label=c,capsize=3)
        ax.axvline(.5,color='black',ls=':',lw=1); ax.set_xlim(0,1.02)
        ax.set_yticks(range(len(methods)),methods); ax.set_title(f'{group.capitalize()} members vs nonmembers')
        ax.set_xlabel('ROC AUC (hierarchical bootstrap 95% interval)')
    axes[0].invert_yaxis(); axes[1].legend(loc='best'); fig.tight_layout(); save(fig,'auc_comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,group in zip(axes,['ordinary','strong']):
        for c,color in [('untrained','#999999'),('trained','#0072B2')]:
            r=[next(x for x in rows if x['condition']==c and x['group']==group and x['method']==f'Affinity p{p}') for p in PREFIXES]
            ax.plot(PREFIXES,[x['auc'] for x in r],'o-',label=c,color=color)
            ax.fill_between(PREFIXES,[x['ci_low'] for x in r],[x['ci_high'] for x in r],alpha=.15,color=color)
        ax.axhline(.5,color='black',ls=':'); ax.set_ylim(0,1.02); ax.set_xticks(PREFIXES)
        ax.set_xlabel('Prefix length (tokens)'); ax.set_title(f'{group.capitalize()} members')
    axes[0].set_ylabel('ROC AUC'); axes[1].legend(); fig.tight_layout(); save(fig,'prefix_affinity')
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    grid=np.linspace(0,1,101)
    for ax,ids,title in [(axes[0],np.arange(80),'Ordinary members'),(axes[1],np.arange(80,100),'Strong members')]:
        for method in ['LM loss','Mean affinity','CNN original gap']:
            curves=[]
            for run in runs:
                a=np.array(run['scores']['trained'][method]); pos=a[ids]; neg=a[100:200]
                ts=np.r_[np.inf,np.sort(np.unique(np.r_[pos,neg]))[::-1],-np.inf]
                fpr=np.array([(neg>=t).mean() for t in ts]); tpr=np.array([(pos>=t).mean() for t in ts])
                # Collapse vertical ROC segments using their upper endpoint.
                unique=np.unique(fpr); upper=np.array([tpr[fpr==u].max() for u in unique])
                curves.append(np.interp(grid,unique,upper))
            ax.plot(grid,np.mean(curves,axis=0),label=method)
        ax.plot([0,1],[0,1],':',color='gray'); ax.set_title(title); ax.set_xlabel('False positive rate')
    axes[0].set_ylabel('True positive rate'); axes[1].legend(); fig.tight_layout(); save(fig,'roc_curves')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--epochs',type=int,default=60); p.add_argument('--samples',type=int,default=16)
    p.add_argument('--output',type=Path,default=Path('results/corrected'))
    args=p.parse_args(); assert args.seeds>=2 and args.samples>=2
    torch.set_num_threads(2); args.output.mkdir(parents=True,exist_ok=True)
    runs=[]
    for seed in range(20261,20261+args.seeds):
        print(f'Running seed {seed}',flush=True)
        r=seed_run(seed,args); runs.append(r)
        (args.output/f'seed_{seed}.json').write_text(json.dumps(r,indent=2))
        print(f'Finished seed {seed}; final LM loss {r["lm_loss_history"][-1]:.4f}',flush=True)
    (args.output/'config.json').write_text(json.dumps({**vars(args),'output':str(args.output),
        'prefixes':PREFIXES,'horizon':HORIZON,'vocab':VOCAB,'torch':torch.__version__,
        'numpy':np.__version__},indent=2))
    rows=summarize(runs,args.output); plots(runs,rows,args.output)
    print('DONE: summaries, paired comparisons, PNG and PDF plots saved.',flush=True)


if __name__=='__main__':
    main()
