"""Report every fresh target, with original Membrain and equal-supervision controls."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_spw_benchmark import FRACTIONS
from membrain_prefix import feature_mask

MAIN=['Membrain original','LOSS','Min-k20%','Zlib','Lowercase','Neighbor','SPW-MIA','Membrain']
CONTROLS=['Probability-only learner','Membrain without embeddings','Membrain curves only','Membrain']
COLORS={'Membrain original':'#b9a1ad','LOSS':'#414141','Min-k20%':'#4c8cb5','Zlib':'#9e7c43',
        'Lowercase':'#a68dbc','Neighbor':'#72a080','SPW-MIA':'#e48a34','Membrain':'#ac315d',
        'Probability-only learner':'#2e9a9a','Membrain without embeddings':'#7b6db2','Membrain curves only':'#caa849'}


def roc(y,s):
    order=np.argsort(-s,kind='stable');values=s[order];y=y[order]
    ends=np.r_[np.flatnonzero(np.diff(values)),len(values)-1]
    return np.r_[0,np.cumsum(1-y)[ends]]/(len(y)-y.sum()),np.r_[0,np.cumsum(y)[ends]]/y.sum()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('results/membrain_prefix'));root=p.parse_args().input
    out=root/'plots';out.mkdir(exist_ok=True)
    records=[]
    for path in sorted((root/'cells').glob('target_*/metrics.json')):
        r=json.loads(path.read_text());r['path']=path.parent;records.append(r)
    if len(records)!=12:raise RuntimeError(f'Need 12 fresh targets; found {len(records)}')
    selection=json.loads((root/'selection.json').read_text())
    scenarios=[(c,e) for c in ['ag_news','wikitext2'] for e in ['standard','repeated']]
    def group(c,e):return [r for r in records if r['corpus']==c and r['regime']==e]
    def finish(fig,name):
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for title,methods,name in [('Fresh-target membership benchmark',MAIN+['Probability-only learner'],'method_comparison'),
                               ('Same-shadow-supervision controls',CONTROLS+['Min-k20%'],'ablations')]:
        fig,axes=plt.subplots(2,2,figsize=(14,8),sharey=True)
        for ax,(c,e) in zip(axes.flat,scenarios):
            for i,m in enumerate(methods):
                vals=np.array([r['metrics'][m]['auc'] for r in group(c,e)]);mean=vals.mean()
                ax.bar(i,mean,color=COLORS[m],width=.75)
                ax.errorbar(i,mean,yerr=np.array([[mean-vals.min()],[vals.max()-mean]]),fmt='none',color='black',capsize=3)
            ax.axhline(.5,ls='--',color='#999999',lw=1);ax.set_ylim(0,1.02);ax.set_ylabel('ROC AUC')
            labels=[m.replace('Probability-only learner','Probability only').replace('Membrain without embeddings','No embeddings').replace('Membrain curves only','Curves only') for m in methods]
            ax.set_xticks(range(len(methods)),labels,rotation=35,ha='right');ax.set_title(f'{c}: {e}')
        fig.suptitle(title+' · means and ranges of three seeds');fig.tight_layout();finish(fig,name)

    methods=['LOSS','Min-k20%','SPW-MIA','Membrain original','Probability-only learner','Membrain']
    fig,axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True);grid=np.linspace(0,1,501)
    for ax,(c,e) in zip(axes.flat,scenarios):
        for m in methods:
            curves=[]
            for r in group(c,e):
                z=np.load(r['path']/'scores.npz');fp,tp=roc(z['labels'],z[m]);curves.append(np.interp(grid,fp,tp))
            ax.plot(grid,np.mean(curves,0),label=m,color=COLORS[m],lw=2 if m=='Membrain' else 1)
        ax.plot([0,1],[0,1],'--',color='#999999');ax.set_title(f'{c}: {e}')
        ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate')
    axes[0,0].legend(fontsize=8);fig.tight_layout();finish(fig,'roc_curves')

    for metric,label in [('weighted_cosine','SPW-weighted aligned embedding cosine'),('accuracy','Generated suffix token accuracy'),
                         ('rollout_excess_ce','Free-running TRUE-token CE minus teacher CE')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
        for ax,(c,e) in zip(axes.flat,scenarios):
            values={'ordinary members':[],'nonmembers':[]}
            if e=='repeated':values['heavily repeated members']=[]
            for r in group(c,e):
                z=np.load(r['path']/'trajectories.npz');v=z[metric]
                values['nonmembers'].append(v[128:].mean(0));values['ordinary members'].append(v[32:128].mean(0) if e=='repeated' else v[:128].mean(0))
                if e=='repeated':values['heavily repeated members'].append(v[:32].mean(0))
            for k,vs in values.items():ax.plot(FRACTIONS,np.mean(vs,0),marker='.',label=k)
            if metric=='accuracy':ax.set_ylim(0,1.02)
            ax.legend(fontsize=7);ax.set_title(f'{c}: {e}');ax.set_ylabel(label);ax.set_xlabel('Fraction of TRUE prefix supplied')
        fig.tight_layout();finish(fig,f'prefix_{metric}')

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,(c,e) in zip(axes.flat,scenarios):
        ps=[];ys=[]
        for r in group(c,e):
            z=np.load(r['path']/'scores.npz');ps.extend(z['Membrain']);ys.extend(z['labels'])
        ps=np.array(ps);ys=np.array(ys);centers=[];actual=[];counts=[]
        ids=np.minimum((ps*10).astype(int),9)
        for j in range(10):
            mask=ids==j
            if mask.any():centers.append(ps[mask].mean());actual.append(ys[mask].mean());counts.append(mask.sum())
        ax.plot([0,1],[0,1],'--',color='#999999');ax.scatter(centers,actual,s=np.sqrt(counts)*8,color=COLORS['Membrain'])
        ax.plot(centers,actual,color=COLORS['Membrain'],lw=1);ax.set_title(f'{c}: {e}')
        ax.set_xlabel('Predicted membership probability');ax.set_ylabel('Observed member fraction')
    fig.suptitle('Shadow-calibrated probabilities on fresh targets · balanced prior only')
    fig.tight_layout();finish(fig,'calibration')

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True);ratios=[.05,.1,.2,.4,.8,1.]
    prevalence=[max(1,round(v*128))/(128+max(1,round(v*128))) for v in ratios]
    for ax,(c,e) in zip(axes.flat,scenarios):
        for m in methods:
            vals=[[r['imbalance'][str(ratio)][m] for ratio in ratios] for r in group(c,e)]
            ax.plot(prevalence,np.mean(vals,0),label=m,color=COLORS[m])
        ax.plot(prevalence,prevalence,'--',color='#999999');ax.set_title(f'{c}: {e}')
        ax.set_xlabel('Member prevalence');ax.set_ylabel('Average precision')
    axes[0,0].legend(fontsize=8);fig.tight_layout();finish(fig,'imbalance')

    fig,ax=plt.subplots(figsize=(11,5))
    vals=[r['calibration']['Membrain']['fixed_threshold_fpr']*100 for r in records]
    ax.bar(range(len(records)),vals,color=COLORS['Membrain']);ax.axhline(1,ls='--',color='#444444',label='1% calibration target')
    ax.set_xticks(range(len(records)),[f'{r["corpus"]}/{r["regime"]}/{r["seed"]}' for r in records],rotation=55,ha='right')
    ax.set_ylabel('Realized false positives on fresh targets (%)');ax.legend();fig.tight_layout();finish(fig,'fixed_threshold_fpr')

    allmethods=MAIN+CONTROLS[:-1];macro={m:{k:float(np.mean([r['metrics'][m][k] for r in records]))
             for k in ['auc','tpr_at_1pct_fpr','average_precision']} for m in allmethods}
    (root/'macro_metrics.json').write_text(json.dumps(macro,indent=2)+'\n')
    rows=[]
    for r in records:
        for m in allmethods:
            v=r['metrics'][m];row=dict(corpus=r['corpus'],regime=r['regime'],seed=r['seed'],method=m,
              auc=v['auc'],auc_ci_low=v['auc_ci95'][0],auc_ci_high=v['auc_ci95'][1],
              tpr_at_1pct_fpr=v['tpr_at_1pct_fpr'],average_precision=v['average_precision'])
            for ref,d in v['differences'].items():
                row.update({f'delta_vs_{ref}':d['delta'],f'delta_ci_low_vs_{ref}':d['ci95'][0],f'delta_ci_high_vs_{ref}':d['ci95'][1]})
            rows.append(row)
    with (root/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Membrain: SPW-inspired revision on fresh benchmarks','',
       'The method is still called **Membrain**. The previous origin-CNN gap is **Membrain original**. '
       'These are new small CPU-model experiments, not the paper’s numerical reproduction.','',
       '## What changed','',
       'Keep increasing true prefixes and whole free-running generated suffixes. Compare aligned hidden states before pooling; '
       'weight positions by the probability change caused by adaptive prefix shortening; learn membership directly on shadow models. '
       'Use separate shadows for validation and probability calibration, and freeze the attack before fresh-target training. '
       'The learner also receives LOSS/Min-k/SPW features, so the same-supervision probability-only control is essential.','',
       '## Fresh-target results','',
       'Two corpora × two exposure regimes × three new target seeds = twelve targets. '
       'Each target evaluates 128 members and 128 nonmembers. All target documents are disjoint from the previous benchmark, '
       'shadow fitting/validation/calibration documents, and origin auxiliaries. Targets share documents across their three seeds; '
       'the twelve targets are not twelve independent datasets. Macro averages give targets equal weight.','',
       '| Method | ROC AUC | TPR at ≤1% FPR | Average precision |','|---|---:|---:|---:|']
    for m in allmethods:
        v=macro[m];lines.append(f'| {m} | {v["auc"]:.4f} | {100*v["tpr_at_1pct_fpr"]:.2f}% | {v["average_precision"]:.4f} |')
    delta=macro['Membrain']['auc']-macro['Membrain original']['auc']
    increment=macro['Membrain']['auc']-macro['Probability-only learner']['auc']
    lines+=['',f'Updated Membrain changes macro AUC by **{delta:+.4f}** relative to the original on the SAME fresh targets. '
       f'The incremental difference from the equally supervised probability-only control is **{increment:+.4f}**. '
       'The earlier reported 0.5847 original-Membrain AUC used different documents and seeds; use the rerun original here for direct comparison.','',
       '| Scenario | Original | LOSS | Min-k20% | SPW-MIA | Probability-only learner | Membrain |','|---|---:|---:|---:|---:|---:|---:|']
    for c,e in scenarios:
        lines.append(f'| {c}/{e} | '+' | '.join(f'{np.mean([r["metrics"][m]["auc"] for r in group(c,e)]):.4f}'
             for m in ['Membrain original','LOSS','Min-k20%','SPW-MIA','Probability-only learner','Membrain'])+' |')
    lines+=['','## Paired uncertainty for the primary Membrain score','',
       'Per-target 95% percentile intervals from 1000 stratified, paired sample bootstraps. These condition on one fitted target, '
       'do not include attack-training uncertainty, and are not corrected for multiple comparisons.','',
       '| Target | Membrain AUC | Δ vs original [95% CI] | Δ vs Min-k [95% CI] | Δ vs probability-only [95% CI] |',
       '|---|---:|---|---|---|']
    for r in records:
        v=r['metrics']['Membrain'];cells=[]
        for ref in ['Membrain original','Min-k20%','Probability-only learner']:
            d=v['differences'][ref];cells.append(f'{d["delta"]:+.4f} [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}]')
        lines.append(f'| {r["corpus"]}/{r["regime"]}/{r["seed"]} | {v["auc"]:.4f} | '+' | '.join(cells)+' |')
    lines+=['','## Does the continuation idea add information?','']
    for ref in ['Membrain original','LOSS','Min-k20%','SPW-MIA','Probability-only learner']:
        ds=[r['metrics']['Membrain']['differences'][ref] for r in records]
        lines.append(f'- Versus {ref}: higher point estimate in {sum(d["delta"]>0 for d in ds)}/12 targets; '
                     f'paired CI entirely above zero in {sum(d["ci95"][0]>0 for d in ds)}/12; '
                     f'entirely below zero in {sum(d["ci95"][1]<0 for d in ds)}/12.')
    lines+=['','Same-learner controls distinguish a better membership objective from evidence that embeddings or generation '
       'add value beyond probability features. A gain over the old origin CNN alone is insufficient to establish a new advantage over '
       'strong membership baselines. Saturated repeated-exposure conditions also leave little room for improvement.','',
       '## Probability calibration and fixed-threshold transfer','',
       'The selected classifier is Platt-calibrated using only separate labeled shadow targets at 50% membership prevalence. '
       'Probabilities require that prior and a similar model/data distribution. They are not validated probabilities for arbitrary web text. '
       'The fixed threshold allows at most floor(0.01×512) false positives among pooled shadow-calibration nonmembers. '
       'Fresh-target FPR can exceed 1%; test thresholds are never retuned.','',
       '| Scenario | Brier score | Log loss | Fixed-threshold TPR | Fixed-threshold FPR |','|---|---:|---:|---:|---:|']
    for c,e in scenarios:
        g=group(c,e);v={k:np.mean([r['calibration']['Membrain'][k] for r in g]) for k in ['brier','log_loss','fixed_threshold_tpr','fixed_threshold_fpr']}
        lines.append(f'| {c}/{e} | {v["brier"]:.4f} | {v["log_loss"]:.4f} | {v["fixed_threshold_tpr"]*100:.2f}% | {v["fixed_threshold_fpr"]*100:.2f}% |')
    lines+=['','## Frozen development choices','',
       '| Variant | Features | Regularization λ |','|---|---:|---:|']
    for m,v in selection.items():lines.append(f'| {m} | {v["features"]} | {v["chosen_regularization"]} |')
    lines+=['','Fit on twelve shadow targets; select on four different shadow targets; refit on fitting+validation targets; '
       'calibrate on four more separate shadow targets. No target sample statistics are used for feature normalization. '
       'The frozen manifest records shadow-role names and attack artifact hashes.','',
       '## Limits and access assumptions','',
       'Membrain now requires labeled shadow training and hidden-state access, unlike the fixed-score probability baselines. '
       'The probability-only learner has the same shadow labels and regularization-selection budget. '
       'Neighbor retains the locally trained masked-model proposal adaptation. Models have 182,016 parameters and 48-token chunks. '
       'The two exposure regimes change both repetition and total training steps. No large-model or external-domain claim follows.',
       '', 'With only 128 nonmembers per target, FPR steps are 0.78125%, so empirical TPR at 1% FPR allows one false positive. '
       'Sample bootstraps and three seeds do not establish reliable population-level low-FPR guarantees. '
       'All methods, all targets, and all predeclared controls are reported, including losses.',
       '', 'See [protocol](../../experiments/MEMBRAIN_PREFIX_PROTOCOL.md) for exact settings and the distinction from SPW-MIA.','',
       '## Figures','']
    for name in ['method_comparison','ablations','roc_curves','prefix_weighted_cosine','prefix_accuracy',
                 'prefix_rollout_excess_ce','calibration','imbalance','fixed_threshold_fpr']:
        lines += [f'![{name}](plots/{name}.png)','']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(macro,indent=2));print('Report:',root/'REPORT.md')


if __name__=='__main__':main()
