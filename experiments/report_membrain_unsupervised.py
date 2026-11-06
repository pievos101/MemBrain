"""Unsupervised Membrain results, including the Min-k anchor and fixed ablations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from membrain_unsupervised import FRACTIONS,recovery_support,corrected_alignment

MAIN=['LOSS','Min-k20%','Zlib','Lowercase','Neighbor','SPW-MIA','Membrain']
ABLATIONS=['Min-k20%','Membrain','Membrain recovery only','Membrain without embeddings',
           'Membrain without rollout loss','Membrain uniform weights']
COLORS=dict(zip(MAIN,['#424242','#438cba','#9e7e43','#9d89b4','#75a181','#e38c36','#ad325e']))


def roc(y,s):
    ids=np.argsort(-s,kind='stable');v=s[ids];y=y[ids];ends=np.r_[np.flatnonzero(np.diff(v)),len(v)-1]
    return np.r_[0,np.cumsum(1-y)[ends]]/(len(y)-y.sum()),np.r_[0,np.cumsum(y)[ends]]/y.sum()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('results/membrain_unsupervised'))
    root=p.parse_args().input;out=root/'plots';out.mkdir(exist_ok=True);records=[]
    for path in sorted((root/'cells').glob('*/metrics.json')):
        r=json.loads(path.read_text());r['path']=path.parent;records.append(r)
    if len(records)!=12:raise RuntimeError(f'Expected12 targets, found{len(records)}')
    scenarios=[(c,e) for c in ['ag_news','wikitext2'] for e in ['standard','repeated']]
    def group(c,e):return [r for r in records if r['corpus']==c and r['regime']==e]
    def finish(fig,name):
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for ax,(c,e) in zip(axes.flat,scenarios):
        for i,m in enumerate(MAIN):
            vals=np.array([r['metrics'][m]['auc'] for r in group(c,e)]);v=vals.mean()
            ax.bar(i,v,color=COLORS[m]);ax.errorbar(i,v,yerr=np.array([[v-vals.min()],[vals.max()-v]]),fmt='none',color='black',capsize=3)
        ax.set_xticks(range(len(MAIN)),MAIN,rotation=35,ha='right');ax.set_ylim(0,1.02)
        ax.axhline(.5,ls='--',color='#999999');ax.set_title(f'{c}: {e}');ax.set_ylabel('ROC AUC')
    fig.suptitle('Unsupervised attacks · fresh targets · mean and range of three seeds')
    fig.tight_layout();finish(fig,'method_comparison')

    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
    for ax,(c,e) in zip(axes.flat,scenarios):
        names=['Min-k20%','Membrain','Recovery only','No embeddings','No rollout loss','Uniform weights']
        means=[np.mean([r['metrics'][m]['auc'] for r in group(c,e)]) for m in ABLATIONS]
        ax.bar(range(len(means)),means,color=['#438cba','#ad325e','#cccccc','#7c6aab','#b49550','#72a080'])
        ax.axhline(.5,ls='--',color='#999999');ax.set_ylim(0,1.02);ax.set_ylabel('ROC AUC')
        ax.set_xticks(range(len(means)),names,rotation=35,ha='right');ax.set_title(f'{c}: {e}')
    fig.suptitle('All fixed ablations · no post-hoc selection of a primary score')
    fig.tight_layout();finish(fig,'ablations')

    fig,ax=plt.subplots(figsize=(10,7));delta=[];lower=[];upper=[];names=[]
    for r in records:
        d=r['metrics']['Membrain']['differences']['Min-k20%'];v=d['delta'];delta.append(v)
        lower.append(v-d['ci95'][0]);upper.append(d['ci95'][1]-v);names.append(f'{r["corpus"]}/{r["regime"]}/{r["seed"]}')
    ax.errorbar(delta,range(len(records)),xerr=[lower,upper],fmt='o',color=COLORS['Membrain'],capsize=3)
    ax.axvline(0,color='#555555',ls='--');ax.set_yticks(range(len(records)),names);ax.invert_yaxis()
    ax.set_xlabel('Membrain AUC minus Min-k20% AUC · paired sample-bootstrap 95% CI')
    fig.tight_layout();finish(fig,'delta_vs_min_k')

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True);grid=np.linspace(0,1,501)
    for ax,(c,e) in zip(axes.flat,scenarios):
        for m in ['LOSS','Min-k20%','SPW-MIA','Membrain']:
            curves=[]
            for r in group(c,e):
                with np.load(r['path']/'scores.npz') as z:fp,tp=roc(z['labels'],z[m])
                curves.append(np.interp(grid,fp,tp))
            ax.plot(grid,np.mean(curves,0),label=m,color=COLORS[m])
        ax.plot([0,1],[0,1],'--',color='#999999');ax.set_title(f'{c}: {e}')
        ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate')
    axes[0,0].legend();fig.tight_layout();finish(fig,'roc_curves')

    for key,label in [('support','Fixed recovery support R'),('alignment','Null-corrected embedding alignment'),
                       ('accuracy','Generated suffix token accuracy')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
        for ax,(c,e) in zip(axes.flat,scenarios):
            groups={'ordinary members':[],'nonmembers':[]}
            if e=='repeated':groups['heavily repeated members']=[]
            for r in group(c,e):
                with np.load(r['path']/'curves.npz') as z:
                    if key=='support':v=recovery_support(z['aligned'],z['null'],z['rollout_ce'],z['teacher_ce'])[0]
                    elif key=='alignment':v=corrected_alignment(z['aligned'],z['null'])
                    else:v=z[key]
                groups['ordinary members'].append(v[32:128].mean(0) if e=='repeated' else v[:128].mean(0))
                groups['nonmembers'].append(v[128:].mean(0))
                if e=='repeated':groups['heavily repeated members'].append(v[:32].mean(0))
            for name,vs in groups.items():ax.plot(FRACTIONS,np.mean(vs,0),marker='.',label=name)
            ax.set_ylim(0,1.02);ax.legend(fontsize=7);ax.set_title(f'{c}: {e}')
            ax.set_xlabel('Fraction of TRUE prefix supplied');ax.set_ylabel(label)
        fig.tight_layout();finish(fig,f'prefix_{key}')

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True);ratios=[.05,.1,.2,.4,.8,1.]
    prevalence=[max(1,round(g*128))/(128+max(1,round(g*128))) for g in ratios]
    for ax,(c,e) in zip(axes.flat,scenarios):
        for m in ['LOSS','Min-k20%','SPW-MIA','Membrain']:
            v=[[r['imbalance'][str(g)][m] for g in ratios] for r in group(c,e)]
            ax.plot(prevalence,np.mean(v,0),label=m,color=COLORS[m])
        ax.plot(prevalence,prevalence,'--',color='#999999');ax.set_title(f'{c}: {e}')
        ax.set_xlabel('Member prevalence');ax.set_ylabel('Average precision')
    axes[0,0].legend();fig.tight_layout();finish(fig,'imbalance')

    methods=MAIN+ABLATIONS[2:];macro={m:{k:float(np.mean([r['metrics'][m][k] for r in records]))
       for k in ['auc','tpr_at_1pct_fpr','average_precision']} for m in methods}
    (root/'macro_metrics.json').write_text(json.dumps(macro,indent=2)+'\n')
    rows=[]
    for r in records:
        for m in methods:
            v=r['metrics'][m];row=dict(corpus=r['corpus'],regime=r['regime'],seed=r['seed'],method=m,
              auc=v['auc'],tpr_at_1pct_fpr=v['tpr_at_1pct_fpr'],average_precision=v['average_precision'],
              auc_ci_low=v['auc_ci95'][0],auc_ci_high=v['auc_ci95'][1])
            for ref,d in v['differences'].items():row.update({f'delta_vs_{ref}':d['delta'],f'delta_ci_low_vs_{ref}':d['ci95'][0],f'delta_ci_high_vs_{ref}':d['ci95'][1]})
            rows.append(row)
    with (root/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    delta=macro['Membrain']['auc']-macro['Min-k20%']['auc']
    lines=['# Membrain — unsupervised membership detection','',
       '**The current Membrain attack is membership-label-free and has no fitted classifier.** '
       'It does not use shadow membership labels, origin training, learned normalization, or membership-probability calibration. '
       'The supervised Membrain experiment is archived separately and its 0.9623 AUC is not attributed to this method.','',
       '## Fixed method','',
       'Retain increasing true prefixes and generate the entire remaining suffix freely. Compare aligned hidden states, '
       'subtract a deterministic shifted-alignment null, and use SPW-inspired token weights from adaptive prefix shortening. '
       'Combine embedding alignment and relative free-running true-token CE degradation into a recovery curve R(r). '
       'Earlier recovery earns more support. The primary score is **Min-k20% + log(early-weighted recovery support)**. '
       'Min-k is an explicit anchor; the independent value of the recovery correction must be tested against it. '
       'The score is computed per candidate, with no batch normalization and no labels. It is NOT a membership probability.',
       '', 'See the [fixed protocol](../../experiments/MEMBRAIN_UNSUPERVISED_PROTOCOL.md) for every formula/constant. '
       'The scoring code/protocol were hashed before target training and not changed after viewing results.','',
       '## Fresh benchmark','',
       'Twelve small target models: two corpora × two exposure regimes × three new seeds. '
       'Use fresh AG News test-source descriptions and WikiText-2 test-source lines, excluding exact chunks from prior '
       'experiments and tokenizer background documents. Source split names do not determine membership: our own target '
       'training randomly assigns it. Each target evaluates 128 members/128 nonmembers. Target pools are shared across '
       'seeds, so twelve models are not twelve independent data populations.','',
       '| Method | Macro ROC AUC | TPR at ≤1% FPR | Average precision |','|---|---:|---:|---:|']
    for m in methods:
        v=macro[m];lines.append(f'| {m} | {v["auc"]:.4f} | {100*v["tpr_at_1pct_fpr"]:.2f}% | {v["average_precision"]:.4f} |')
    lines+=['',f'Membrain’s macro AUC difference from Min-k20% is **{delta:+.4f}**. '
       'A high hybrid AUC alone is not evidence that recovery improves on the likelihood anchor. '
       'All ablations keep their predefined names/formulas/signs; none becomes the primary method after evaluation.','',
       '| Scenario | LOSS | Min-k20% | SPW-MIA | Membrain | Recovery only |','|---|---:|---:|---:|---:|---:|']
    for c,e in scenarios:
        lines.append(f'| {c}/{e} | '+' | '.join(f'{np.mean([r["metrics"][m]["auc"] for r in group(c,e)]):.4f}'
          for m in ['LOSS','Min-k20%','SPW-MIA','Membrain','Membrain recovery only'])+' |')
    lines+=['','## Paired differences from the Min-k anchor','',
       '1000 paired stratified sample bootstraps per fitted target. Intervals do not include target-training '
       'uncertainty, are not multiplicity-adjusted, and cannot be treated as independent across shared documents.','',
       '| Target | Membrain AUC | Δ vs Min-k20% | Paired 95% CI |','|---|---:|---:|---|']
    for r in records:
        v=r['metrics']['Membrain'];d=v['differences']['Min-k20%']
        lines.append(f'| {r["corpus"]}/{r["regime"]}/{r["seed"]} | {v["auc"]:.4f} | {d["delta"]:+.4f} | [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}] |')
    lines+=['','## Interpretation','']
    for ref in ['LOSS','Min-k20%','SPW-MIA']:
        ds=[r['metrics']['Membrain']['differences'][ref] for r in records]
        lines.append(f'- Versus {ref}: higher AUC in {sum(d["delta"]>0 for d in ds)}/12 targets; '
            f'paired CI above zero in {sum(d["ci95"][0]>0 for d in ds)}/12 and below zero in {sum(d["ci95"][1]<0 for d in ds)}/12.')
    lines+=['','Heavily repeated texts may show much earlier embedding/token recovery, but this does not imply '
       'a universal direction for the rollout-versus-teacher loss gap. For ordinary members, teacher-forced loss can '
       'improve while a single greedy rollout still diverges, increasing that gap. This is a limitation of the fixed '
       'recovery hypothesis, not a reason to flip its sign after observing the labels.',
       '', 'Membership labels enter only target-model training and metric computation. The saved candidate-only '
       'scores precede metric computation. Benchmark labels do not train the attack. No operating threshold is '
       'fitted; converting scores to calibrated probabilities or a guaranteed false-positive rate would require '
       'additional assumptions/data and is not done here.',
       '', '## Limits','',
       '182,016-parameter causal Transformers, 48-token samples, three seeds, shared document pools. No large-model '
       'or external-domain result follows. Repeated exposure also changes total training steps. Exact deduplication '
       'does not rule out semantic/partial overlap. Membrain needs hidden-state access; the probability baselines do '
       'not. Neighbor retains its documented local masked-LM proposal adaptation.',
       '', 'At 128 negatives per target, FPR steps are 0.78125%; empirical TPR at 1% permits only one false positive. '
       'This is retrospective evaluation, not deployment calibration. Prior supervised and old unsupervised numbers '
       'used different documents/seeds and cannot be read as paired comparisons.','', '## Figures','']
    for name in ['method_comparison','ablations','delta_vs_min_k','roc_curves','prefix_support','prefix_alignment','prefix_accuracy','imbalance']:
        lines += [f'![{name}](plots/{name}.png)','']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(macro,indent=2));print('Report:',root/'REPORT.md')


if __name__=='__main__':main()
