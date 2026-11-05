"""Natural-sentence recovery plots, conditional uncertainty, and a sentence viewer."""
import csv
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/workspace/.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import membrain_recovery as m
from membrain_loss_curves import auc

OUT=Path(__file__).resolve().parents[1]/'results/membrain_recovery'
COLORS=['#777777','#0072B2','#E69F00','#009E73']


def main():
    records=[];generations=[]
    config=json.loads((OUT/'config.json').read_text())
    texts=json.loads((OUT/'sentences.json').read_text())['evaluation']
    for seed in config['seeds']:
        with np.load(OUT/f'model_{seed}.npz') as f:records.append({k:f[k] for k in f.files})
        generations.append(json.loads((OUT/f'generations_{seed}.json').read_text())['greedy_suffixes'])
    keys=['suffix_exact','tail_exact','suffix_accuracy','tail_accuracy','suffix_cosine','tail_cosine']
    cubes={k:np.empty((4,128,len(m.FRACTIONS),config['samples']+1)) for k in keys}
    seen=np.zeros((4,128),int);examples=[[None]*128 for _ in range(4)]
    model_ids=np.empty((4,128),int)
    for i,r in enumerate(records):
        assert np.array_equal(r['fractions'],m.FRACTIONS)
        for level in range(4):
            ids=np.where(r['repeats']==m.LEVELS[level])[0]
            seen[level,ids]+=1;model_ids[level,ids]=config['seeds'][i]
            for key in keys:cubes[key][level,ids]=r[key][ids]
            for doc in ids:examples[level][doc]=[gen[doc] for gen in generations[i]]
    assert np.all(seen==1),'Every sentence must occur at every exposure exactly once.'
    counts=m.LEVELS*config['epochs'];rng=np.random.default_rng(7821)
    bootstrap=rng.integers(0,128,(1000,128))
    def curve(values):
        mu=values.mean(0);bs=values[bootstrap].mean(1)
        lo,hi=np.quantile(bs,[.025,.975],axis=0)
        return mu,lo,hi
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    def draw(ax,key,sampled):
        for level,color in enumerate(COLORS):
            values=cubes[key][level,:,:,1:].mean(2) if sampled else cubes[key][level,:,:,0]
            mu,lo,hi=curve(values)
            ax.plot(m.FRACTIONS,mu,'o-',color=color,label=f'{counts[level]} presentations',ms=4)
            ax.fill_between(m.FRACTIONS,lo,hi,color=color,alpha=.13)
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.set_xlabel('Fraction of true sentence supplied as prefix')
        ax.set_ylim(-.02,1.02);ax.set_xlim(0,.92)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,key,sampled,title in [(axes[0,0],'suffix_exact',False,'Whole remaining sentence: greedy'),
        (axes[0,1],'suffix_exact',True,'Whole remaining sentence: sampling'),
        (axes[1,0],'tail_exact',False,'Fixed final 8 characters: greedy'),
        (axes[1,1],'tail_exact',True,'Fixed final 8 characters: sampling')]:
        draw(ax,key,sampled);ax.set_title(title)
        ax.set_ylabel('Mean sampled exact-match rate' if sampled else 'Fraction of sentences exactly recovered')
    axes[0,0].legend(fontsize=9);fig.suptitle('Membrain: does more exposure reduce the prefix needed for recovery?',y=1.02)
    save(fig,'recovery_curves')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key,title in [(axes[0],'suffix_exact','Whole remaining sentence'),(axes[1],'tail_exact','Fixed final 8 characters')]:
        draw(ax,key,False);ax.set_xlim(0,.205);ax.set_xticks([.01,.05,.1,.15,.2]);ax.tick_params(axis='x',labelsize=9)
        ax.set_title(f'Greedy recovery: {title.lower()}');ax.set_ylabel('Fraction of sentences exactly recovered')
    axes[1].legend(fontsize=8);save(fig,'early_prefix_zoom')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for ax,key,title in [(axes[0,0],'suffix_accuracy','Whole suffix: token accuracy'),
                        (axes[0,1],'tail_accuracy','Fixed final segment: token accuracy'),
                        (axes[1,0],'suffix_cosine','Whole suffix: pooled hidden-state cosine'),
                        (axes[1,1],'tail_cosine','Fixed final segment: pooled hidden-state cosine')]:
        draw(ax,key,True);ax.set_title(title)
        ax.set_ylabel('Character accuracy' if 'accuracy' in key else 'Cosine similarity')
        if 'cosine' in key:ax.set_ylim(.5,1.01)
    axes[0,0].legend(fontsize=8);save(fig,'gradual_recovery')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key,title in [(axes[0],'suffix_cosine','Whole remaining sentence'),(axes[1],'tail_cosine','Fixed final 8 characters')]:
        draw(ax,key,False);ax.set_ylim(.5,1.01);ax.set_title(f'Greedy embedding similarity: {title.lower()}')
        ax.set_ylabel('Pooled hidden-state cosine')
    axes[0].legend(fontsize=8);save(fig,'embedding_recovery')
    threshold_rows=[];thresholds={}
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,sampled in zip(axes,[False,True]):
        for level,color in enumerate(COLORS):
            values=cubes['suffix_exact'][level,:,:,1:].mean(2) if sampled else cubes['suffix_exact'][level,:,:,0]
            threshold=m.sustained_threshold(values);thresholds[level,sampled]=threshold
            cdf=(threshold[:,None]<=m.FRACTIONS).mean(0)
            ax.step(m.FRACTIONS,cdf,where='post',color=color,label=f'{counts[level]} presentations')
            median=next((f for f,v in zip(m.FRACTIONS,cdf) if v>=.5),None)
            threshold_rows.append(dict(presentations=int(counts[level]),decoding='sampled' if sampled else 'greedy',
                median_sustained_prefix=median if median is not None else '>90%',
                censored_fraction=float(np.isinf(threshold).mean())))
        ax.set_title('Sampling: estimated success ≥50%' if sampled else 'Greedy: exact recovery')
        ax.set_xlabel('Sustained recovery threshold (prefix fraction)');ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(0,1.02)
    axes[0].set_ylabel('Fraction of sentences recovered by this threshold');axes[1].legend(fontsize=8)
    save(fig,'recovery_thresholds')
    # Numerical curves and conditional paired exposure effects.
    rows=[]
    for key in keys:
        for sampled in [False,True]:
            for level in range(4):
                values=cubes[key][level,:,:,1:].mean(2) if sampled else cubes[key][level,:,:,0]
                mu,lo,hi=curve(values)
                for j,f in enumerate(m.FRACTIONS):rows.append(dict(metric=key,decoding='sampled' if sampled else 'greedy',
                    presentations=int(counts[level]),prefix_fraction=f,mean=mu[j],ci_low=lo[j],ci_high=hi[j]))
    effects=[]
    for key in ['suffix_exact','tail_exact']:
        for f in [.05,.1,.5,.9]:
            j=np.where(np.isclose(m.FRACTIONS,f))[0][0]
            d=cubes[key][3,:,j,0]-cubes[key][0,:,j,0]
            lo,hi=np.quantile(d[bootstrap].mean(1),[.025,.975])
            effects.append(dict(metric=key,prefix_fraction=f,comparison=f'{counts[3]} minus 0 presentations',
                                difference=d.mean(),ci_low=lo,ci_high=hi))
    for name,data in [('curve_summary',rows),('threshold_summary',threshold_rows),('paired_exposure_effects',effects)]:
        with (OUT/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    # A descriptive comparison with LOSS, without fitting or selecting classifiers.
    auc_rows=[]
    for level in m.LEVELS[1:]:
        for method in ['Greedy recovery area','Sampled recovery area','Sampled tail accuracy area','Sampled embedding similarity area','LOSS full sentence']:
            values=[]
            for r in records:
                if method=='Greedy recovery area':score=r['suffix_exact'][:,:,0].mean(1)
                elif method=='Sampled recovery area':score=r['suffix_exact'][:,:,1:].mean((1,2))
                elif method=='Sampled tail accuracy area':score=r['tail_accuracy'][:,:,1:].mean((1,2))
                elif method=='Sampled embedding similarity area':score=r['suffix_cosine'][:,:,1:].mean((1,2))
                else:
                    if 'teacher_full_loss' not in r:
                        lm=m.SentenceLM();lm.load_state_dict(torch_load(OUT/f'model_{int(r["seed"])}.pt'))
                        x,lengths=m.encode(texts)
                        import torch
                        with torch.no_grad():
                            loss=torch.nn.functional.cross_entropy(lm(x[:,:-1]).transpose(1,2),x[:,1:],ignore_index=0,reduction='none')
                            r['teacher_full_loss']=(loss.sum(1)/(lengths-1)).numpy()
                    score=-r['teacher_full_loss']
                values.append(auc(score[r['repeats']==level],score[r['repeats']==0]))
            auc_rows.append(dict(presentations=int(level*config['epochs']),method=method,
                                 mean_auc=np.mean(values),model_sd=np.std(values,ddof=1)))
    with (OUT/'descriptive_auc.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(auc_rows[0]));w.writeheader();w.writerows(auc_rows)
    assert all(np.array_equal(r['prefix_sizes'],records[0]['prefix_sizes']) for r in records)
    viewer(texts,examples,model_ids,counts,records[0]['prefix_sizes'])
    text='''# Membrain: prefix-triggered whole-sentence recovery

## Main finding

This controlled experiment supports Membrain's prefix-triggered recall hypothesis:
the heavily exposed sentences can be recovered from much shorter prefixes.
After 640 presentations, greedy generation reproduces the entire remainder for
81.2% of sentences from a 5% prefix and 88.3% from a 10% prefix. The median
sustained greedy recovery threshold is 2%. At 160 presentations, only 20.3%
recover even at 90% prefix; unseen and 40-presentation sentences have no exact
recoveries on this grid. The fixed-final-segment control shows the same greedy
exposure effect, so shortening the scored remainder does not fully explain it.

Sampling is much less reliable than greedy decoding: even after 640 presentations,
the mean exact-match rate at a 90% prefix is about 49%. The result demonstrates
a strong-exposure recall mechanism in a small character model, not a calibrated
membership attack or a general advantage over LOSS.

This experiment directly tests the hypothesis that repeated exposure lets a model
recover a sentence from a shorter prefix. Every prefix is followed by generation
of the **entire remaining sentence**, not a fixed eight-token continuation.

![Whole-sentence and fixed-tail recovery](recovery_curves.png)
![Short-prefix detail](early_prefix_zoom.png)
![Recovery thresholds](recovery_thresholds.png)

## Design

The dataset contains 128 real Shakespeare sentences, each 80–120 characters long,
and 128 disjoint background sentences. Sentences are segmented at terminal
punctuation, whitespace normalized, and deduplicated before selection. A fixed
printable-ASCII vocabulary is used; 0 is a PAD token. Speaker headers at the start
of a candidate are removed. Source and checksum match the retained Tiny Shakespeare
artifact in `../membrain_loss_curves/source/`.

Four independently initialized character GRUs have 32-dimensional token embeddings
and 96 hidden units. Each receives ten background-language epochs, then 40 epochs
with controlled evaluation-sentence repetition factors 0, 1, 4 and 16. Thus the
actual full-sentence presentation counts are **0, 40, 160 and 640**. Zero-exposure
sentences never occur in that model's background or sentence training sets.
Common words and substrings may occur in other sentences.

A balanced four-model rotation gives every evaluation sentence each exposure
level exactly once. Every model has 32 sentences per exposure level. This controls
sentence identity across the aggregate exposure comparison; model weights and
other sentences' exposures differ between rotations. It is not a single-model
counterfactual. The selected sentences and all model weights are retained.

Prefixes cover 1%, 2%, 5%, then 10% through 90% of each true sentence. Rounded
character counts are used. The 1/2/5% grid was added after observing recovery
already high at 10% for heavily repeated sentences; only decoding was rerun,
using the same saved weights. This is an exploratory refinement, not a
preregistered threshold grid.

For each prefix, the model generates until the known true sentence length. The
prefix is true text; every subsequent input is generated text. No true suffix
characters are fed back. Generation is evaluated both greedily and using 16
sampled paths at temperature 1.0. PAD is masked from generation. Knowing the
remaining length simplifies the task; end-of-sentence stopping is not tested.

## Measurements

- **Whole-suffix exact recovery:** every generated remainder character matches.
- **Character accuracy:** fraction of matching positions in the remainder.
- **Fixed-final-segment recovery:** match the final eight characters after freely
  generating the entire bridge from the supplied prefix. Those eight true target
  characters are the same at every prefix and are never included in a supplied
  prefix. This controls the shortening of the whole-suffix task.
- **Embedding similarity:** cosine similarity between mean-pooled causal GRU
  states for the true and generated suffixes, also measured on the fixed final
  segment. These are model-internal representations, not an independent semantic
  encoder. High cosine alone does not establish correct recall or membership.

A sustained recovery threshold is the smallest tested prefix that succeeds at
that point and every later tested prefix. Greedy success is exact match; sampled
success means at least 50% of the 16 paths exactly match. Thresholds not reached
by 90% remain censored; they are not converted into successes or omitted.

Bands on recovery plots are 95% bootstrap intervals over the 128 sentence IDs,
paired across exposure levels and conditional on these four fitted models. They
do not capture uncertainty from training a new collection of LMs. Sampled paths
are averaged within sentences, not treated as independent sentences. Thresholds
from 16 samples have substantial Monte Carlo uncertainty.

## Numerical recovery

| Training presentations | Greedy exact at 5% prefix | Greedy exact at 10% | Greedy exact at 50% | Greedy exact at 90% | Sampled exact at 90% |
|---:|---:|---:|---:|---:|---:|
'''
    for i,count in enumerate(counts):
        points=[]
        for f in [.05,.1,.5,.9]:
            j=np.where(np.isclose(m.FRACTIONS,f))[0][0];points.append(cubes['suffix_exact'][i,:,j,0].mean())
        sampled=cubes['suffix_exact'][i,:,-1,1:].mean()
        text+=f'| {count} | '+' | '.join(f'{p:.1%}' for p in points)+f' | {sampled:.1%} |\n'
    text+='\n### Sustained recovery thresholds\n\n| Presentations | Decoding | Median threshold | Not recovered by 90% |\n|---:|---|---:|---:|\n'
    for r in threshold_rows:
        median=r['median_sustained_prefix'];label=median if isinstance(median,str) else f'{median:.0%}'
        text+=f'| {r["presentations"]} | {r["decoding"]} | {label} | {r["censored_fraction"]:.1%} |\n'
    text+='''
## Reading the result

The primary question is whether heavily exposed sentences recover at shorter
prefixes, and whether this persists for the fixed final segment. Greedy and
sampling curves answer different questions: a model can rank the correct path
first yet sample it infrequently over a long remainder. Exact-match probabilities
multiply across positions, so longer sampled suffixes are especially demanding.
Do not equate low sampled exact-match rates with absence of learned recall.

The embedding curves are a secondary diagnostic. Shared language structure,
pooling and representation anisotropy can produce high cosine for wrong text.
Compare their exposure effects with exact recovery and character accuracy rather
than interpreting absolute cosine as a probability of membership.

`descriptive_auc.csv` compares fixed, untrained recovery summaries with negative
whole-sentence teacher loss for each seen exposure level versus unseen sentences.
It reports the mean and SD across four models, not a calibrated attack or an
independent validation of classifier superiority. No score is tuned using these
evaluation outcomes. Recovery-area labels refer to an equally weighted mean
over the tested grid, which has extra resolution at short prefixes, not a
continuous integral.

## Limits

This is a deliberately strong memorization positive control with a small
character-level GRU, a small sentence pool and up to 640 presentations. It can
support the proposed recovery mechanism in this setting; it does not establish
performance on naturally occurring training membership or large pretrained LMs.
Four models and one source corpus do not establish broad robustness. Exact
deduplication does not remove similar sentences. Prefixes and target lengths
are known. The final eight-character control may include predictable endings.
No membership classifier is fitted in this experiment: it first measures the
recovery signal that Membrain seeks to capture.

## Explore and reproduce

[Open the sentence-by-sentence recovery viewer](sentence_viewer.html). It shows
all 128 sentences, every prefix, and all four exposure conditions, with generated
character errors highlighted. Each row comes from a different rotation model;
the seed is shown. Examples are not filtered to successful outputs.

Figures have PNG and PDF versions: `recovery_curves`, `early_prefix_zoom`,
`gradual_recovery`, `embedding_recovery`, and `recovery_thresholds`. CSVs contain curve summaries,
thresholds, paired exposure effects and descriptive LOSS comparisons. Raw metrics,
all greedy generations, sentences, weights and configuration are retained.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/workspace/.venvs/membrain/bin/python -m unittest discover -s experiments -p test_membrain_recovery.py
/workspace/.venvs/membrain/bin/python experiments/membrain_recovery.py
/workspace/.venvs/membrain/bin/python experiments/report_membrain_recovery.py
```

Run from the repository root with Python 3.12, NumPy 2.2.6, PyTorch 2.6.0+cpu
and Matplotlib 3.10.8. Existing matching results are reused; use a fresh `--output`
for changed training settings. Prefix-grid refinements use retained LM weights.
The report generator reads the default result directory. Earlier experiments
and the six original benchmark files are unchanged.
'''
    (OUT/'REPORT.md').write_text(text)
    print('Saved Membrain recovery plots, tables, report and interactive sentence viewer.')


def torch_load(path):
    import torch
    return torch.load(path,weights_only=True)


def viewer(texts,examples,model_ids,counts,prefix_sizes):
    data=json.dumps(dict(texts=texts,examples=examples,models=model_ids.tolist(),
                         counts=counts.tolist(),fractions=m.FRACTIONS.tolist())).replace('<',r'\u003c')
    html='''<!doctype html><html><head><meta charset="utf-8"><title>Membrain sentence recovery</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;background:#fafafa;color:#222}
section{background:white;border:1px solid #ddd;border-radius:8px;padding:18px;margin:14px 0}
.text{font:18px/1.7 monospace;white-space:pre-wrap}.wrong{background:#ffd8d8;color:#9b111e}.right{color:#167348}.prefix{color:#0072b2;font-weight:bold}
label{display:block;margin:16px 0}input{width:60%}select{max-width:100%;padding:6px}</style></head><body>
<h1>Membrain: how much prefix triggers recovery?</h1><p>The same sentence at four training exposure levels. Each row uses a different rotation model; all 128 sentences were tested at every exposure. Greedy decoding generates the entire remaining sentence.</p>
<label>Sentence <select id="sentence"></select></label><label>Prefix <input id="prefix" type="range" min="0" max="11" value="3"> <b id="fraction"></b></label>
<section><b>True sentence</b><div id="truth" class="text"></div></section><div id="results"></div>
<p>Blue text is supplied. Red characters differ from the true continuation; green characters match. The final eight target characters are never supplied. These are controlled character-GRU results, not membership probabilities.</p>
<script>const data=DATA;
const sel=document.getElementById('sentence'),slider=document.getElementById('prefix');
data.texts.forEach((t,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+': '+t;sel.appendChild(o)});
function span(t,cls){let s=document.createElement('span');s.textContent=t;s.className=cls;return s}
function render(){let i=+sel.value,k=+slider.value,t=data.texts[i],n=Math.round(t.length*data.fractions[k]);
// Python uses ties-to-even; all relevant displayed prefix lengths are supplied below.
n=data.prefixSizes[i][k];document.getElementById('fraction').textContent=Math.round(100*data.fractions[k])+'% ('+n+' characters)';
let truth=document.getElementById('truth');truth.replaceChildren(span(t.slice(0,n),'prefix'),span(t.slice(n),''));
let root=document.getElementById('results');root.replaceChildren();data.counts.forEach((count,l)=>{let section=document.createElement('section'),h=document.createElement('b');
let g=data.examples[l][i][k],correct=g===t.slice(n),tail=g.slice(-8)===t.slice(-8);h.textContent=count+' presentations · model '+data.models[l][i]+' · full suffix '+(correct?'EXACT':'not exact')+' · final 8 '+(tail?'EXACT':'not exact');
section.appendChild(h);let d=document.createElement('div');d.className='text';d.appendChild(span(t.slice(0,n),'prefix'));
Array.from(g).forEach((c,j)=>d.appendChild(span(c,c===t[n+j]?'right':'wrong')));section.appendChild(d);root.appendChild(section)})}
slider.max=data.fractions.length-1;sel.onchange=render;slider.oninput=render;render();</script></body></html>'''
    # Store actual rounded prefix sizes so the viewer exactly matches Python.
    parsed=json.loads(data)
    parsed['prefixSizes']=prefix_sizes.tolist()
    data=json.dumps(parsed).replace('<',r'\u003c')
    (OUT/'sentence_viewer.html').write_text(html.replace('DATA',data))


if __name__=='__main__':main()
