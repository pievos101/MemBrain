"""Results and actual two-embedding trajectories; all examples predetermined."""
import json
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from membrain_unsupervised import FRACTIONS

ROOT=Path('results/membrain_simple')
MAIN=['LOSS','Min-k20%','Zlib','Lowercase','Neighbor','SPW-MIA','Previous Membrain','Membrain']
CONTROLS=['Embedding curve','Teacher loss curve','Rollout loss curve','Membrain without rollout weighting',
          'Loss curves without embeddings','Membrain']


def main():
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT);ROOT=p.parse_args().input
    out=ROOT/'plots';out.mkdir(exist_ok=True);records=[]
    for p in sorted((ROOT/'cells').glob('*/metrics.json')):
        r=json.loads(p.read_text());r['path']=p.parent;records.append(r)
    if len(records)!=12:raise RuntimeError(f'Need12 completed cells, found{len(records)}')
    scenarios=[(c,e) for c in ['ag_news','wikitext2'] for e in ['standard','repeated']]
    def group(c,e):return [r for r in records if r['corpus']==c and r['regime']==e]
    def finish(fig,name):
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for methods,title,name in [(MAIN[:2]+['SPW-MIA','Previous Membrain','Teacher loss curve','Loss curves without embeddings','Membrain'],
        'Fixed unsupervised scores on fresh texts','comparison'),(CONTROLS,'Predeclared trajectory/loss ablations','ablations')]:
        fig,axes=plt.subplots(2,2,figsize=(14,8),sharey=True)
        for ax,(c,e) in zip(axes.flat,scenarios):
            for i,m in enumerate(methods):
                vals=np.array([r['metrics'][m]['auc'] for r in group(c,e)]);v=vals.mean()
                ax.bar(i,v,color='#ac315d' if m=='Membrain' else '#538ca9')
                ax.errorbar(i,v,yerr=np.array([[v-vals.min()],[vals.max()-v]]),fmt='none',color='black',capsize=3)
            ax.set_xticks(range(len(methods)),methods,rotation=35,ha='right');ax.set_ylim(0,1.02)
            ax.axhline(.5,color='#999999',ls='--');ax.set_title(f'{c}: {e}');ax.set_ylabel('ROC AUC')
        fig.suptitle(title+' · means and ranges of three seeds');fig.tight_layout();finish(fig,name)

    for key,label in [('cosine','True/generated suffix embedding cosine'),('teacher_ce','Teacher-forced TRUE-suffix CE'),
                       ('rollout_ce','Free-running CE against TRUE suffix tokens'),('accuracy','Generated suffix token accuracy')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
        for ax,(c,e) in zip(axes.flat,scenarios):
            values={'ordinary members':[],'nonmembers':[]}
            if e=='repeated':values['heavily repeated members']=[]
            for r in group(c,e):
                with np.load(r['path']/'trajectories.npz') as z:v=z[key]
                values['ordinary members'].append(v[32:128].mean(0) if e=='repeated' else v[:128].mean(0))
                values['nonmembers'].append(v[128:].mean(0))
                if e=='repeated':values['heavily repeated members'].append(v[:32].mean(0))
            for k,vs in values.items():ax.plot(FRACTIONS,np.mean(vs,0),marker='.',label=k)
            if key=='accuracy':ax.set_ylim(0,1.02)
            ax.legend(fontsize=7);ax.set_title(f'{c}: {e}');ax.set_xlabel('Fraction of TRUE prefix supplied');ax.set_ylabel(label)
        fig.tight_layout();finish(fig,f'prefix_{key}')

    fig,axes=plt.subplots(2,2,figsize=(11,9))
    for row,corpus in enumerate(['ag_news','wikitext2']):
        r=next(r for r in records if r['corpus']==corpus and r['seed']==6101 and r['regime']=='standard')
        with np.load(r['path']/'trajectories.npz') as z:a,b=z['true_embedding'],z['generated_embedding']
        a=a/np.maximum(np.linalg.norm(a,axis=2,keepdims=True),1e-12);b=b/np.maximum(np.linalg.norm(b,axis=2,keepdims=True),1e-12)
        joined=np.r_[a.reshape(-1,a.shape[-1]),b.reshape(-1,b.shape[-1])];mu=joined.mean(0)
        _,singular,vt=np.linalg.svd(joined-mu,full_matrices=False);basis=vt[:2].T
        variance=(singular[:2]**2)/(singular**2).sum()
        for col,idx in enumerate([0,128]):
            ax=axes[row,col]
            for values,name,color in [(a,'Class 0: true suffix','#3679ae'),(b,'Class 1: generated suffix','#db893c')]:
                points=(values[idx]-mu)@basis;ax.plot(points[:,0],points[:,1],'-o',label=name,color=color,markersize=4)
                for j in [0,4,9]:ax.annotate(f'{FRACTIONS[j]:.2f}',points[j],fontsize=8)
            ax.set_title(f'{corpus}, seed6101, {"member row0" if idx==0 else "nonmember row128"}')
            ax.set_xlabel(f'PC1 ({variance[0]:.0%})');ax.set_ylabel(f'PC2 ({variance[1]:.0%})');ax.legend(fontsize=7)
    fig.suptitle('Two suffix-embedding trajectories · fixed examples · PCA visualization only')
    fig.tight_layout();finish(fig,'two_embedding_trajectories')

    methods=list(dict.fromkeys(MAIN+CONTROLS));macro={m:{k:float(np.mean([r['metrics'][m][k] for r in records]))
       for k in ['auc','tpr_at_1pct_fpr','average_precision']} for m in methods}
    (ROOT/'macro_metrics.json').write_text(json.dumps(macro,indent=2)+'\n')
    rows=[]
    for r in records:
        for m in methods:
            v=r['metrics'][m];rows.append(dict(corpus=r['corpus'],regime=r['regime'],seed=r['seed'],method=m,
               auc=v['auc'],tpr_at_1pct_fpr=v['tpr_at_1pct_fpr'],average_precision=v['average_precision'],
               auc_ci_low=v['auc_ci95'][0],auc_ci_high=v['auc_ci95'][1],
               delta_vs_teacher=v['differences']['Teacher loss curve']['delta'],
               delta_vs_teacher_ci_low=v['differences']['Teacher loss curve']['ci95'][0],
               delta_vs_teacher_ci_high=v['differences']['Teacher loss curve']['ci95'][1]))
    with (ROOT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Membrain: two embedding curves and absolute continuation loss','',
      'The simplified attack is still **Membrain**, and remains membership-label-free with no fitted attack classifier. '
      'The primary method has no Min-k anchor, SPW token weights, shifted-null correction, or rollout-minus-teacher loss gap.','',
      '## The two signals and the missing link','',
      'At each increasing true prefix, retain one mean hidden vector for the true suffix and one for the freely generated suffix. '
      'Call them origin class 0 and class 1 if useful. They are two known origins, not membership classes. '
      'A true/generated classifier can learn generic generation artifacts and achieve strong origin separation while being '
      'uninformative about membership. We therefore compare the paired vectors directly and preserve the entire prefix curves.',
      '', 'The missing check is confidence in the SPECIFIC true continuation. Embedding proximity can mean that the generated '
      'text is plausible or generic. Teacher-forced true-token loss measures specific conditional fit; absolute free-running '
      'true-token loss measures reproduction under the model’s own earlier outputs. The previous relative gap could increase '
      'for members when conditional fit improved faster than greedy recovery.',
      '', '## Fixed simple score','',
      '`d(r) = (1 - cosine(z_true(r), z_generated(r))) / 2`',
      '', '**Membrain = −A[teacher_loss(r) + d(r) × rollout_loss(r)].**',
      '', 'A is a fixed normalized integral over ten prefix fractions .05 through .9, weighted by 1-r to reward earlier recovery. '
      'Rollout loss always targets TRUE tokens while conditioning on generated history. Both embeddings pool suffix tokens '
      'only, after those tokens are processed. Higher score means stronger membership evidence. This heuristic is not a '
      'calibrated probability, and no membership labels choose coefficients, score signs, normalization, or thresholds.',
      '', '## Fresh experiments','',
      'Twelve targets: AG News and WikiText-2, standard/repeated exposure, three new seeds 6101/6102/6103. '
      '384 new documents per corpus, excluding all previous exact chunks and tokenizer background. Each target evaluates '
      '128 members and 128 nonmembers. The target models are the same 182,016-parameter causal Transformers as before. '
      'Seeds share document pools; they are not twelve independent data populations. The full protocol was fixed before training.',
      '', '| Method | Macro ROC AUC | TPR at ≤1% FPR | Average precision |','|---|---:|---:|---:|']
    for m in methods:
        v=macro[m];lines.append(f'| {m} | {v["auc"]:.4f} | {100*v["tpr_at_1pct_fpr"]:.2f}% | {v["average_precision"]:.4f} |')
    lines+=['','## Does adding the two embedding curves help?','',
      f'Primary Membrain minus teacher-loss-curve AUC: **{macro["Membrain"]["auc"]-macro["Teacher loss curve"]["auc"]:+.4f}**. '
      f'Primary minus Min-k20%: **{macro["Membrain"]["auc"]-macro["Min-k20%"]["auc"]:+.4f}**. '
      'These controls matter: good membership detection does not automatically establish incremental information in embeddings.',
      '', '| Target | Membrain AUC | Δ vs teacher curve | Paired 95% CI |','|---|---:|---:|---|']
    for r in records:
        v=r['metrics']['Membrain'];d=v['differences']['Teacher loss curve']
        lines.append(f'| {r["corpus"]}/{r["regime"]}/{r["seed"]} | {v["auc"]:.4f} | {d["delta"]:+.4f} | [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}] |')
    lines+=['','All controls are predeclared and all are reported. None replaces the primary after evaluation. '
      'The embedding-only and loss-only ablations identify where the information resides; the loss-curves-without-embeddings '
      'control tests whether apparent gains need the vectors. Paired intervals are 1000-draw stratified sample bootstraps '
      'conditional on each fitted target, without training-uncertainty or multiplicity correction.',
      '', '## What class 0/class 1 can mean','',
      'Automatically labeling true suffix paths class 0 and generated paths class 1 gives a self-supervised origin task. '
      'It does not supply member/nonmember labels. Turning an origin classifier’s uncertainty into membership evidence '
      'requires a validated relationship; otherwise it can be confidently wrong for membership. This experiment stays '
      'unfitted and directly tests the proposed origin-convergence link, with loss as the additional observable.',
      '', '## Limits','',
      'Small 48-token CPU tests, shared pools across seeds, and stronger hidden-state access for Membrain than for '
      'probability-only baselines. Repeated exposure changes both repetition and optimization budget. Exact chunk exclusion '
      'does not eliminate semantic overlap. At 128 negatives, FPR steps are 0.78125%, so empirical TPR at 1% allows one false '
      'positive; no deployment threshold or membership probability is fitted. These are new benchmarks, not a replication '
      'of billion-parameter paper numbers. Prior studies informed the hypothesis; this is not external preregistration.',
      '', 'Saved vectors are 64-dimensional. The two-dimensional PCA examples use the same per-model projection for both '
      'origins, without membership labels, solely for visualization. Example indices 0 and 128 were fixed before evaluation. '
      'The score always uses full-dimensional cosine, never the plotted coordinates.',
      '', 'See [protocol](../../experiments/MEMBRAIN_SIMPLE_PROTOCOL.md).','', '## Plots','']
    for name in ['comparison','ablations','two_embedding_trajectories','prefix_cosine','prefix_teacher_ce','prefix_rollout_ce','prefix_accuracy']:
        lines.extend([f'![{name}](plots/{name}.png)',''])
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(macro,indent=2))


if __name__=='__main__':main()
