"""Complete report of predeclared trajectory-geometry hypotheses and loss controls."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from membrain_shape import cos
from membrain_unsupervised import FRACTIONS

PRIMARY='Membrain shape hybrid'
MAIN=['LOSS','Min-k20%','SPW-MIA','Previous Membrain','Hard-token teacher loss curve',PRIMARY]
GEOMETRY=['Membrain uniform geometry','Membrain hard-token geometry','Uniform proximity','Hard-token proximity',
          'Uniform centered','Hard-token centered','Uniform velocity','Hard-token velocity']


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('results/membrain_shape'))
    root=p.parse_args().input;out=root/'plots';out.mkdir(exist_ok=True);rs=[]
    for path in sorted((root/'cells').glob('*/metrics.json')):
        r=json.loads(path.read_text());r['path']=path.parent;rs.append(r)
    if len(rs)!=12:raise RuntimeError(f'Expected12 targets; found{len(rs)}')
    scenarios=[(c,e) for c in ['ag_news','wikitext2'] for e in ['standard','repeated']]
    def group(c,e):return [r for r in rs if r['corpus']==c and r['regime']==e]
    def finish(fig,name):
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for methods,title,name in [(MAIN,'Likelihood controls and the fixed geometry hybrid','comparison'),
                                (GEOMETRY,'Geometry alone: all predefined component scores','geometry_ablations')]:
        fig,axes=plt.subplots(2,2,figsize=(14,8),sharey=True)
        for ax,(c,e) in zip(axes.flat,scenarios):
            for i,m in enumerate(methods):
                vals=np.array([r['metrics'][m]['auc'] for r in group(c,e)]);v=vals.mean()
                ax.bar(i,v,color='#ac315d' if m==PRIMARY else '#558ba5')
                ax.errorbar(i,v,yerr=np.array([[v-vals.min()],[vals.max()-v]]),fmt='none',color='black',capsize=3)
            labels=[m.replace('Membrain ','').replace('Hard-token teacher loss curve','Hard-token loss curve') for m in methods]
            ax.set_xticks(range(len(methods)),labels,rotation=35,ha='right');ax.set_ylim(0,1.02)
            ax.axhline(.5,ls='--',color='#aaaaaa');ax.set_title(f'{c}: {e}');ax.set_ylabel('ROC AUC')
        fig.suptitle(title+' · mean and range of three seeds');fig.tight_layout();finish(fig,name)

    for prefix,title in [('', 'Ordinary pooled suffix embeddings'),('hard_','Hard-token pooled suffix embeddings')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
        for ax,(c,e) in zip(axes.flat,scenarios):
            values={'ordinary members':[],'nonmembers':[]}
            if e=='repeated':values['heavily repeated members']=[]
            for r in group(c,e):
                with np.load(r['path']/'trajectories.npz') as z:a,b=z[prefix+'true_embedding'],z[prefix+'generated_embedding']
                a=a/np.maximum(np.linalg.norm(a,axis=2,keepdims=True),1e-10);b=b/np.maximum(np.linalg.norm(b,axis=2,keepdims=True),1e-10)
                a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True);agreement=cos(a,b)
                values['ordinary members'].append(agreement[32:128].mean(0) if e=='repeated' else agreement[:128].mean(0))
                values['nonmembers'].append(agreement[128:].mean(0))
                if e=='repeated':values['heavily repeated members'].append(agreement[:32].mean(0))
            for label,v in values.items():ax.plot(FRACTIONS,np.mean(v,0),marker='.',label=label)
            ax.set_ylim(-1,1);ax.legend(fontsize=7);ax.set_title(f'{c}: {e}');ax.set_xlabel('True-prefix fraction');ax.set_ylabel('Centered trajectory agreement')
        fig.suptitle(title);fig.tight_layout();finish(fig,'centered_curves_'+('hard' if prefix else 'uniform'))

    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for row,c in enumerate(['ag_news','wikitext2']):
        r=next(r for r in rs if r['corpus']==c and r['seed']==7101 and r['regime']=='standard')
        with np.load(r['path']/'trajectories.npz') as z:a,b=z['hard_true_embedding'],z['hard_generated_embedding']
        a=a/np.maximum(np.linalg.norm(a,axis=2,keepdims=True),1e-10);b=b/np.maximum(np.linalg.norm(b,axis=2,keepdims=True),1e-10)
        a-=a.mean(1,keepdims=True);b-=b.mean(1,keepdims=True)
        joined=np.r_[a.reshape(-1,64),b.reshape(-1,64)];mu=joined.mean(0);_,s,vt=np.linalg.svd(joined-mu,full_matrices=False)
        for col,idx in enumerate([0,128]):
            ax=axes[row,col]
            for values,label in [(a,'True suffix'),(b,'Generated suffix')]:
                points=(values[idx]-mu)@vt[:2].T;ax.plot(points[:,0],points[:,1],'-o',label=label,markersize=4)
                for j in [0,4,9]:ax.annotate(f'{FRACTIONS[j]:.2f}',points[j],fontsize=8)
            ax.set_title(f'{c}, {"member row0" if idx==0 else "nonmember row128"}');ax.legend(fontsize=7)
            ax.set_xlabel(f'PC1 ({s[0]**2/(s*s).sum():.0%})');ax.set_ylabel(f'PC2 ({s[1]**2/(s*s).sum():.0%})')
    fig.suptitle('Centered hard-token trajectories · fixed rows, seed7101 · PCA display only')
    fig.tight_layout();finish(fig,'two_centered_trajectories')

    controls=[]
    for r in rs:
        path=r['path']/'tail_control_metrics.json'
        if path.exists():controls.append(json.loads(path.read_text()))
    if controls and len(controls)!=12:raise RuntimeError('Finish all12 exploratory tail controls')
    if controls:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
        for ax,(c,e) in zip(axes.flat,scenarios):
            values={'ordinary members':[],'nonmembers':[]}
            if e=='repeated':values['heavily repeated members']=[]
            for r in group(c,e):
                with np.load(r['path']/'tail_control.npz') as z:v=z['cosine']
                values['ordinary members'].append(v[32:128].mean(0) if e=='repeated' else v[:128].mean(0))
                values['nonmembers'].append(v[128:].mean(0))
                if e=='repeated':values['heavily repeated members'].append(v[:32].mean(0))
            for label,v in values.items():ax.plot(FRACTIONS,np.mean(v,0),marker='.',label=label)
            ax.legend(fontsize=7);ax.set_title(f'{c}: {e}');ax.set_xlabel('True-prefix fraction');ax.set_ylabel('Fixed-tail true/generated cosine')
        fig.suptitle('Exploratory control: compare the SAME final four token positions')
        fig.tight_layout();finish(fig,'fixed_tail_recovery')

    fig,ax=plt.subplots(figsize=(10,7));deltas=[];low=[];high=[];labels=[]
    for r in rs:
        d=r['metrics'][PRIMARY]['differences']['Min-k20%'];v=d['delta'];deltas.append(v)
        low.append(v-d['ci95'][0]);high.append(d['ci95'][1]-v);labels.append(f'{r["corpus"]}/{r["regime"]}/{r["seed"]}')
    ax.errorbar(deltas,range(len(rs)),xerr=[low,high],fmt='o',color='#ac315d',capsize=3)
    ax.axvline(0,color='#777777',ls='--');ax.set_yticks(range(len(rs)),labels);ax.invert_yaxis()
    ax.set_xlabel('Hybrid AUC minus Min-k20% · paired sample-bootstrap 95% CI');fig.tight_layout();finish(fig,'delta_vs_min_k')

    methods=list(rs[0]['metrics']);macro={m:{k:float(np.mean([r['metrics'][m][k] for r in rs]))
           for k in ['auc','tpr_at_1pct_fpr','average_precision']} for m in methods}
    (root/'macro_metrics.json').write_text(json.dumps(macro,indent=2)+'\n')
    rows=[]
    for r in rs:
        for m,v in r['metrics'].items():
            d=v['differences']['Min-k20%'];rows.append(dict(corpus=r['corpus'],regime=r['regime'],seed=r['seed'],method=m,
             auc=v['auc'],tpr_at_1pct_fpr=v['tpr_at_1pct_fpr'],average_precision=v['average_precision'],
             auc_ci_low=v['auc_ci95'][0],auc_ci_high=v['auc_ci95'][1],delta_vs_min_k=d['delta'],delta_ci_low=d['ci95'][0],delta_ci_high=d['ci95'][1]))
    with (root/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Membrain experiments: trajectory shape and difficult tokens','',
     '**All scoring is membership-label-free, with no fitted attack classifier or calibrated membership probability.** '
     'This is a held-out hypothesis test; no best-performing exploratory score replaces the predefined primary.','',
     '## What we tested','',
     'Keep one true-suffix and one generated-suffix embedding per increasing prefix. Test whether a generic origin offset '
     'hides matching trajectory shape: normalize vectors, center each candidate’s trajectory over prefixes, and compare '
     'the centered paths and their step directions. Separately pool only the hardest 20% of suffix positions, selected '
     'by teacher-forced true-token loss; pool BOTH origins at the SAME selected positions. This injects loss information '
     'into which embeddings are compared, without fitting on membership labels.',
     '', 'Fixed geometry score = mean(proximity, centered correlation, velocity agreement). '
     '**Primary candidate: Min-k20% + 0.5 × hard-token geometry.** The anchor is explicit; high hybrid AUC by itself '
     'cannot demonstrate that embeddings add value. Geometry-only variants, components, path-length diagnostics, and '
     'loss-only curves are all reported. Candidate normalization uses only that candidate, never other test samples.',
     '', '## Fresh evaluation','',
     'Twelve tiny target models: two corpora, two exposure regimes, three new seeds 7101/7102/7103. '
     '768 new exact-deduplicated documents exclude all earlier experiment chunks and tokenizer background. '
     'Each target evaluates 128 members and 128 nonmembers. Seeds share corpus pools, so these are not twelve '
     'independent document populations. Target architecture/training match the preceding suites.',
     '', '| Method | Macro ROC AUC | TPR at ≤1% FPR | Average precision |','|---|---:|---:|---:|']
    for m in methods:
        v=macro[m];lines.append(f'| {m} | {v["auc"]:.4f} | {100*v["tpr_at_1pct_fpr"]:.2f}% | {v["average_precision"]:.4f} |')
    delta=macro[PRIMARY]['auc']-macro['Min-k20%']['auc']
    lines+=['',f'Primary hybrid minus Min-k20% macro AUC = **{delta:+.4f}**. '
     'This is the relevant incremental comparison, rather than the hybrid’s absolute AUC.','',
     '| Scenario | Min-k20% | Hard-token loss curve | Uniform geometry | Hard-token geometry | Primary hybrid |',
     '|---|---:|---:|---:|---:|---:|']
    for c,e in scenarios:
        lines.append(f'| {c}/{e} | '+' | '.join(f'{np.mean([r["metrics"][m]["auc"] for r in group(c,e)]):.4f}'
          for m in ['Min-k20%','Hard-token teacher loss curve','Membrain uniform geometry','Membrain hard-token geometry',PRIMARY])+' |')
    lines+=['','## Paired uncertainty','',
     '| Target | Hybrid AUC | Δ vs Min-k20% | Paired 95% CI |','|---|---:|---:|---|']
    for r in rs:
        v=r['metrics'][PRIMARY];d=v['differences']['Min-k20%']
        lines.append(f'| {r["corpus"]}/{r["regime"]}/{r["seed"]} | {v["auc"]:.4f} | {d["delta"]:+.4f} | [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}] |')
    ds=[r['metrics'][PRIMARY]['differences']['Min-k20%'] for r in rs]
    lines+=['',f'Higher hybrid AUC in {sum(d["delta"]>0 for d in ds)}/12 targets; paired CI entirely positive in '
      f'{sum(d["ci95"][0]>0 for d in ds)}/12 and entirely negative in {sum(d["ci95"][1]<0 for d in ds)}/12.',
     '', 'All intervals use 1000 paired stratified sample bootstraps conditional on each fitted target; they omit '
     'training uncertainty and multiple-comparison correction. FPR resolution is 1/128=.78125%; the retrospective '
     '1% metric permits only one false positive. No operating threshold is calibrated.',
     '', '## Interpretation limits','',
     'A true/generated origin difference does not automatically identify training membership. Centering removes an '
     'offset, but generic language regularities can still align trajectories; a member can also fail greedy recovery. '
     'Hard-token pooling is label-free but uses target probabilities, so improvements must be compared to loss-only '
     'controls. Finding a strong component on this test does not validate a newly selected primary on the same data.',
     '', 'Models have 182,016 parameters and 48-token samples. Repeated exposure also changes optimization steps. '
     'Exact exclusion does not remove semantic overlap. These results do not establish billion-parameter or external-domain '
     'performance. Hidden-state access remains stronger than the probability-only baselines. All old trained attack '
     'classifiers are unused. The prior studies motivated this design; this is not an externally preregistered study.',
     '', 'See [protocol](../../experiments/MEMBRAIN_SHAPE_PROTOCOL.md). Plotted PCA uses a shared, unlabeled per-target '
     'projection for visualization only. Fixed example rows 0 and 128 are not chosen for favorable outcomes; scoring '
     'uses full-dimensional geometry.','']
    if controls:
        movement=max(c['maximum_true_embedding_change'] for c in controls)
        change=max(c['maximum_teacher_loss_change'] for c in controls)
        lines+=['','## Exploratory fixed-tail control: what the true curve measures','',
          'Added after preliminary primary results; separately frozen before computing any tail scores. '
          'This control is exploratory and does not replace the original held-out primary. '
          'Use the SAME final four token positions at every prefix, while still generating the whole remaining suffix.',
          '',f'Across ALL candidates and targets, maximum change in the true-tail embedding is **{movement:.8g}**; '
          f'maximum true-tail teacher-loss change is **{change:.8g}**.',
          '', 'True prefix plus true continuation is the same original text at every partition. Thus fixed-position '
          'teacher-forced states are constant. The earlier true-suffix trajectory moves because the pooled token window '
          'shrinks. That movement includes changing content/composition, not additional model exposure to the same target. '
          'Generated-tail recovery still changes with supplied context; comparing it to a fixed true reference isolates '
          'that aspect more cleanly. This does not prove the moving-window trajectories contain no useful information.',
          '', '| Exploratory four-token score | Macro ROC AUC |','|---|---:|']
        for m in controls[0]['metrics']:
            lines.append(f'| {m} | {np.mean([c["metrics"][m]["auc"] for c in controls]):.4f} |')
        lines+=['','These scores use a four-token target instead of the full suffix; do not treat them as a directly '
          'matched recovery-accuracy comparison. No best control is promoted or retuned. '
          'See [control protocol](../../experiments/MEMBRAIN_TAIL_CONTROL.md).','']
    lines+=['','## Plots','']
    names=['comparison','geometry_ablations','centered_curves_uniform','centered_curves_hard','two_centered_trajectories','delta_vs_min_k']
    if controls:names.append('fixed_tail_recovery')
    for name in names:
        lines.extend([f'![{name}](plots/{name}.png)',''])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(macro,indent=2))


if __name__=='__main__':main()
