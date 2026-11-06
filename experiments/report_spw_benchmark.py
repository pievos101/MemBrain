"""Figures and numerical report from completed, independently trained cells."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_spw_benchmark import FRACTIONS

METHODS=['LOSS','Min-k20%','Zlib','Lowercase','Neighbor','SPW-MIA','Membrain']
COLORS=dict(zip(METHODS,['#444444','#4786b4','#a17a45','#a08bc1','#6b9f78','#e58d39','#b13e63']))


def roc(y,s):
    order=np.argsort(-s,kind='stable');v=s[order];y=y[order]
    ends=np.r_[np.flatnonzero(np.diff(v)),len(v)-1]
    tp=np.r_[0,np.cumsum(y)[ends]]/y.sum();fp=np.r_[0,np.cumsum(1-y)[ends]]/(len(y)-y.sum())
    return fp,tp


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('results/spw_scale'))
    root=p.parse_args().input;out=root/'plots';out.mkdir(exist_ok=True)
    records=[]
    for path in sorted(root.glob('*/metrics.json')):
        r=json.loads(path.read_text());r['path']=path.parent;records.append(r)
    if not records:raise RuntimeError('No completed benchmark cells')
    scenarios=[(c,e) for c in ['ag_news','wikitext2'] for e in ['standard','repeated']]
    def group(c,e):return [r for r in records if r['corpus']==c and r['regime']==e]
    def finish(fig,name):
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
    for ax,(c,e) in zip(axes.flat,scenarios):
        g=group(c,e)
        for i,m in enumerate(METHODS):
            vals=[r['metrics'][m]['auc'] for r in g]
            if not vals:continue
            mean=np.mean(vals);ax.bar(i,mean,color=COLORS[m],width=.75)
            ax.errorbar(i,mean,yerr=np.array([[mean-min(vals)],[max(vals)-mean]]),color='black',capsize=3,fmt='none')
        ax.axhline(.5,color='#999999',ls='--',lw=1);ax.set_ylim(0,1)
        ax.set_xticks(range(len(METHODS)),METHODS,rotation=35,ha='right')
        ax.set_title(f'{c}: {e} exposure');ax.set_ylabel('ROC AUC')
    fig.suptitle('Own CPU benchmark · bars: seed mean; whiskers: seed range')
    fig.tight_layout();finish(fig,'auc_comparison')

    fig,axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True)
    grid=np.linspace(0,1,501)
    for ax,(c,e) in zip(axes.flat,scenarios):
        g=group(c,e)
        for m in METHODS:
            curves=[]
            for r in g:
                z=np.load(r['path']/'scores.npz');fp,tp=roc(z['labels'],z[m])
                curves.append(np.interp(grid,fp,tp))
            if curves:ax.plot(grid,np.mean(curves,0),label=m,color=COLORS[m],lw=2 if m in ['Membrain','LOSS','SPW-MIA'] else 1)
        ax.plot([0,1],[0,1],'--',color='#aaaaaa',lw=1);ax.set_title(f'{c}: {e}')
        ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate')
    axes[0,0].legend(fontsize=8);fig.suptitle('ROC curves averaged across training seeds')
    fig.tight_layout();finish(fig,'roc_curves')

    for metric,label in [('accuracy','Generated suffix token accuracy'),('exact','Entire generated suffix exactly correct'),('cosine','True/generated suffix embedding cosine'),
                         ('rollout_ce','Free-running CE against TRUE continuation')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
        for ax,(c,e) in zip(axes.flat,scenarios):
            all_values={'ordinary members':[],'nonmembers':[]}
            if e=='repeated':all_values['repeated members']=[]
            for r in group(c,e):
                z=np.load(r['path']/'trajectories.npz');y=z['labels']
                if metric=='cosine':
                    a,b=z['true'],z['generated'];values=np.sum(a*b,2)/(np.linalg.norm(a,axis=2)*np.linalg.norm(b,axis=2))
                else:values=z[metric]
                all_values['nonmembers'].append(values[y==0].mean(0))
                all_values['ordinary members'].append(values[32:128].mean(0) if e=='repeated' else values[y==1].mean(0))
                if e=='repeated':all_values['repeated members'].append(values[:32].mean(0))
            for name,vs in all_values.items():
                if vs:ax.plot(FRACTIONS,np.mean(vs,0),marker='.',label=name)
            ax.set_title(f'{c}: {e}');ax.set_xlabel('Fraction of true prefix supplied');ax.set_ylabel(label)
            if metric in ['accuracy','exact']:ax.set_ylim(0,1.02)
            ax.legend(fontsize=8)
        fig.tight_layout();finish(fig,f'prefix_{metric}')

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,(c,e) in zip(axes.flat,scenarios):
        gaps={0:[],1:[]}
        for r in group(c,e):
            z=np.load(r['path']/'trajectories.npz');gap=np.abs(z['qtrue']-z['qgenerated'])
            for label in gaps:gaps[label].extend(gap[z['labels']==label])
        bins=np.linspace(0,1,21)
        for label,name in [(0,'nonmembers'),(1,'members')]:
            ax.hist(gaps[label],bins=bins,density=True,histtype='step',linewidth=2,label=name)
        ax.set_title(f'{c}: {e}');ax.set_xlabel('|q(true) − q(generated)|');ax.set_ylabel('Density')
    axes[0,0].legend();fig.suptitle('Origin-probability gaps: overlap limits membership classification')
    fig.tight_layout();finish(fig,'origin_probability_gaps')

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,(c,e) in zip(axes.flat,scenarios):
        g=group(c,e);ratios=[.05,.1,.2,.4,.8,1.]
        prevalence=np.array([max(1,round(x*128))/(128+max(1,round(x*128))) for x in ratios])
        for m in METHODS:
            vals=[[r['imbalances'][str(gamma)][m]['mean_ap'] for gamma in ratios] for r in g]
            if vals:ax.plot(prevalence,np.mean(vals,0),label=m,color=COLORS[m])
        ax.plot(prevalence,prevalence,'--',color='#aaaaaa',label='Chance prevalence')
        ax.set_title(f'{c}: {e}');ax.set_xlabel('Member prevalence');ax.set_ylabel('Average precision')
    axes[0,0].legend(fontsize=8);fig.tight_layout();finish(fig,'membership_imbalance')

    fig,ax=plt.subplots(figsize=(10,5));names=list(records[0]['component_ablations'])
    for i,(c,e) in enumerate(scenarios):
        vals=[np.mean([r['component_ablations'][m]['auc'] for r in group(c,e)]) for m in names]
        ax.plot(range(len(names)),vals,marker='.',label=f'{c}/{e}')
    ax.axhline(.5,color='#aaaaaa',ls='--');ax.set_xticks(range(len(names)),names,rotation=25,ha='right')
    ax.set_ylabel('ROC AUC, seed mean');ax.legend(fontsize=8);fig.tight_layout();finish(fig,'spw_component_ablations')

    rows=[]
    for r in records:
        for m in METHODS:
            v=r['metrics'][m];rows.append(dict(corpus=r['corpus'],regime=r['regime'],seed=r['seed'],method=m,
                **{k:v[k] for k in ['auc','tpr_at_1pct_fpr','average_precision','auprc_trapezoid']},
                auc_ci_low=v['auc_ci95'][0],auc_ci_high=v['auc_ci95'][1],
                delta_auc_vs_loss=v['auc']-r['metrics']['LOSS']['auc'],
                delta_ci_low=v['auc_difference_from_loss_ci95'][0],delta_ci_high=v['auc_difference_from_loss_ci95'][1]))
    with (root/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    macro={m:{k:float(np.mean([r['metrics'][m][k] for r in records])) for k in ['auc','tpr_at_1pct_fpr','average_precision']} for m in METHODS}
    (root/'macro_metrics.json').write_text(json.dumps(macro,indent=2)+'\n')
    lines=['# New CPU-scale membership benchmark results','',
       '**Independent benchmarks, not a numerical reproduction of the supplied paper.**','',
       f'Completed {len(records)} target cells; {len(set(r["seed"] for r in records))} training seeds. '
       '128 members and 128 nonmembers per cell. Equal-weight macro averages; no pooled raw-score ranking.','',
       '| Method | Macro ROC AUC | TPR at ≤1% FPR | Average precision |',
       '|---|---:|---:|---:|']
    for m,v in macro.items():lines.append(f'| {m} | {v["auc"]:.4f} | {100*v["tpr_at_1pct_fpr"]:.2f}% | {v["average_precision"]:.4f} |')
    lines+=['','## Scenario ROC AUC (mean across seeds)','','| Scenario | '+' | '.join(METHODS)+' |','|---|'+'---:|'*len(METHODS)]
    for c,e in scenarios:
        g=group(c,e)
        if g:lines.append(f'| {c}/{e} | '+' | '.join(f'{np.mean([r["metrics"][m]["auc"] for r in g]):.4f}' for m in METHODS)+' |')
    lines+=['','## Membrain versus LOSS: paired, per-target uncertainty','',
      'AUC differences and 95% paired bootstrap intervals (1000 stratified draws). '
      'Intervals condition on the fitted target and sampled documents.','',
      '| Corpus | Exposure | Seed | Δ AUC | 95% interval | Origin holdout AUC |',
      '|---|---|---:|---:|---|---:|']
    for r in records:
        v=r['metrics']['Membrain'];lo,hi=v['auc_difference_from_loss_ci95'];delta=v['auc']-r['metrics']['LOSS']['auc']
        lines.append(f'| {r["corpus"]} | {r["regime"]} | {r["seed"]} | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {r["origin_holdout"]["auc"]:.4f} |')
    differences=[r['metrics']['Membrain']['auc']-r['metrics']['LOSS']['auc'] for r in records]
    positive=sum(r['metrics']['Membrain']['auc_difference_from_loss_ci95'][0]>0 for r in records)
    lines+=['',f'Membrain exceeds LOSS numerically in {sum(x>0 for x in differences)}/{len(records)} cells; '
       f'{positive}/{len(records)} paired intervals lie entirely above zero. This does not establish generalization beyond these targets.',
       '', 'The central distinction is visible in the prefix-recovery curves: heavily repeated members can be recovered more accurately, '
       'but a CNN trained only to separate true and generated trajectories can solve that origin task using patterns common to both '
       'members and nonmembers. Its probability gap then overlaps across membership classes. Origin discrimination is not an objective '
       'that guarantees preservation of the recovery-versus-prefix signal.',
       '', '## Interpretation limits','',
       'Membrain needs hidden-state access and a fitted auxiliary origin classifier. Its score is not a calibrated membership probability. '
       'A strong origin classifier alone does not imply accurate membership inference. Neighbor uses locally trained masked-LM proposals. '
       'The tiny causal models have limited language ability; an embedding convergence signal can reflect generic token patterns as well as memorization.',
       '', 'The 128-negative evaluation resolves FPR in 0.78125% steps; the empirical 1% metric permits only one false positive. '
       'Two seeds and sample bootstrap intervals do not provide robust population-level low-FPR guarantees. '
       'Repeated exposure also changes optimization budget. No membership-score sign was flipped after evaluation.',
       '', 'See [protocol](../../experiments/SPW_BENCHMARK.md) for exact choices and omitted original-paper settings.','']
    lines+=['','## Secondary Membrain diagnostics (not primary-score selection)','',
       '| Scenario | True-origin probability AUC | Signed gap AUC | Direct cosine AUC |',
       '|---|---:|---:|---:|']
    for c,e in scenarios:
        g=group(c,e)
        if g:lines.append(f'| {c}/{e} | '+' | '.join(f'{np.mean([r["secondary_metrics"][m]["auc"] for r in g]):.4f}'
           for m in ['Membrain true-origin score','Membrain signed convergence','Membrain cosine'])+' |')
    lines+=['','## Compute and diagnostics','',
       '| Target | Parameters | Origin holdout AUC | Mean shortened/original prefix | SPW seconds | Membrain seconds |',
       '|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["corpus"]}/{r["regime"]}/{r["seed"]} | {r["parameter_count"]} | '
           f'{r["origin_holdout"]["auc"]:.3f} | {r["mean_shortened_ratio"]:.3f} | '
           f'{r["spw_seconds"]:.2f} | {r["membrain_seconds"]:.2f} |')
    lines+=['','Timings are local wall-clock diagnostics, not a controlled efficiency comparison: SPW time excludes the '
       'already computed full-context teacher pass; Membrain time includes auxiliary trajectory generation and origin fitting. '
       'Neither includes target training.','', '## Figures','']
    for name in ['auc_comparison','roc_curves','prefix_accuracy','prefix_exact','prefix_cosine','prefix_rollout_ce',
                 'origin_probability_gaps','membership_imbalance','spw_component_ablations']:
        lines+= [f'![{name}](plots/{name}.png)','']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(macro,indent=2));print('Report:',root/'REPORT.md')


if __name__=='__main__':main()
