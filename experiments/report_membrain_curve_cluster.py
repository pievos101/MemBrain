"""Report the exploratory, cached-transcript full-curve comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from membrain_curve_cluster import GRID

ROOT=Path('results/membrain_curve_cluster')


def main():
    paths=sorted((ROOT/'cells').glob('*/metrics.json'))
    if len(paths)!=16:raise ValueError('Expected all 16 targets')
    rows=[json.loads(p.read_text()) for p in paths]
    names=list(rows[0]['metrics'])
    summary={m:dict(auc=float(np.mean([r['metrics'][m]['auc'] for r in rows])),
        selected_components={str(k):sum(r['groups'][m]['components']==k for r in rows) for k in (1,2,3)})
        if m in rows[0]['groups'] else dict(auc=float(np.mean([r['metrics'][m]['auc'] for r in rows]))) for m in names}
    (ROOT/'macro_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Membrain: full recovery curves, unsupervised clustering',
      '', 'Exploratory reanalysis of 16 cached small-model targets; zero new model queries. '
      'Each mixture fits 128 unlabeled discovery sentences and is evaluated on 128 separate sentences. '
      'These data were inspected in earlier experiments: this is not fresh confirmation.', '',
      'Eight whole-suffix word-edit distances are retained per sentence. A second curve contains '
      'next-word mismatch frequencies from 16 returned completions per prefix. '
      'Diagonal Gaussian mixtures choose one/two components using unlabeled BIC. '
      'The lower-distance prototype names the high-recovery group. One component means abstention; '
      'its AUC is 0.5. Scores are group posteriors, not calibrated membership probabilities.', '',
      '| Method | Mean held-out membership AUC | Targets selecting 1 / 2 / 3 groups |',
      '|---|---:|---:|']
    for m,v in summary.items():
        counts=v.get('selected_components');count=' / '.join(str(counts[str(k)]) for k in (1,2,3)) if counts else 'raw ranking'
        lines.append(f"| {m} | {v['auc']:.4f} | {count} |")
    lines+=['','Matched scalar controls use the same resampled curves, discovery split, mixture family and fitting settings.', '',
      '| Capacity / exposure | Whole curve | Whole early scalar | Next-word curve | Next-word early scalar | Joint curves |',
      '|---|---:|---:|---:|---:|---:|']
    selected=['Whole curve','Whole early scalar','Next-word curve','Next-word early scalar','Joint curves']
    for capacity in ['small','larger']:
        for regime in ['standard','repeated']:
            subset=[r for r in rows if r['capacity']==capacity and r['regime']==regime]
            values=[np.mean([r['metrics'][m]['auc'] for r in subset]) for m in selected]
            lines.append('| '+capacity+' / '+regime+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines+=['','## Paired per-target comparisons','','| Curve vs matched early scalar | Mean AUC difference | Targets with bootstrap CI above / below zero |',
       '|---|---:|---:|']
    for m,ref in [('Whole curve','Whole early scalar'),('Next-word curve','Next-word early scalar'),('Joint curves','Next-word early scalar')]:
        differences=[r['metrics'][m]['differences'][ref] for r in rows]
        lines.append(f"| {m} vs {ref} | {np.mean([d['delta'] for d in differences]):+.4f} | {sum(d['ci95'][0]>0 for d in differences)} / {sum(d['ci95'][1]<0 for d in differences)} |")
    lines+=['','Intervals use 1,000 paired stratified sentence bootstraps conditional on each fitted model and cached query set. '
      'They omit fit/query/training uncertainty and multiplicity adjustment. Targets share some sentences.', '',
      '## Interpretation','',
      'Curve shape contains information beyond an average, but clustering recovery patterns does not by itself identify training membership. '
      'Read the standard-exposure rows separately from repeated exposure. Report all controls rather than selecting a winner on these labels. '
      'No default detector is promoted from this exploratory comparison.', '',
      'Diagonal covariance ignores correlations between prefix distances; BIC under this approximation can overselect components. '
      'The joint input has 16 coordinates but only 128 discovery examples. Whole-suffix difficulty also changes as the suffix shrinks. '
      'These tiny local models establish no performance claim for commercial or billion-parameter models.', '',
      'Per-target metrics, group precision/recall, high-exposure AUC, bootstrap intervals, frozen source hashes, '
      'unlabeled features, fitted mixtures and held-out scores are retained in this directory.', '',
      '![Held-out comparison](comparison.png)', '', '![Fitted full curves](prototypes.png)']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(selected));width=.19
    for j,(capacity,regime) in enumerate([(c,r) for c in ['small','larger'] for r in ['standard','repeated']]):
        subset=[r for r in rows if r['capacity']==capacity and r['regime']==regime]
        ax.bar(x+(j-1.5)*width,[np.mean([r['metrics'][m]['auc'] for r in subset]) for m in selected],width,label=f'{capacity}, {regime}')
    ax.axhline(.5,color='black',ls=':',lw=1);ax.set(ylim=(.35,1),ylabel='Held-out membership ROC AUC',title='Membrain: full curves vs matched scalar mixtures (exploratory)')
    ax.set_xticks(x,selected,rotation=12);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for ax,(capacity,regime) in zip(axes.flat,[(c,r) for c in ['small','larger'] for r in ['standard','repeated']]):
        path=ROOT/'cells'/f'{capacity}_ag_news_9101_{regime}'/'unlabeled_models.json'
        model=json.loads(path.read_text())['models']['Whole curve']
        for j,(mu,var) in enumerate(zip(model['means'],model['variances'])):
            mu=np.array(mu);sd=np.sqrt(var)
            ax.plot(GRID,mu,marker='o',label=f'Component {j+1} (weight {model["weights"][j]:.2f})')
            ax.fill_between(GRID,np.maximum(0,mu-sd),np.minimum(1,mu+sd),alpha=.12)
        ax.set(title=f'{capacity}, {regime}: {model["components"]} group(s)',ylim=(0,1.05),xlabel='Fraction of available prefix',ylabel='Whole-suffix word-edit distance');ax.legend(fontsize=7)
    fig.suptitle('Unlabeled discovery prototypes: AG News, pool 9101\nBands: diagonal component standard deviation');fig.tight_layout();fig.savefig(ROOT/'prototypes.png',dpi=160);plt.close(fig)
    print((ROOT/'REPORT.md').read_text())


if __name__=='__main__':main()
