"""Prospective confirmation report: every capacity/pool/regime, including failures."""
import argparse
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from membrain_text import words
from membrain_word_probe import score_record

PRIMARY='Membrain 16'
METHODS=[PRIMARY,'Membrain replicate16','Membrain pooled32','Membrain nested3','Hard-prefix word probe',
         'Greedy next-word','Greedy whole-suffix','Greedy fixed-tail','Length-only control','LOSS','Min-k20%','SPW-MIA']
PRIVILEGED={'LOSS','Min-k20%','SPW-MIA'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('results/membrain_text_validation'));root=p.parse_args().input
    out=root/'plots';out.mkdir(exist_ok=True);rs=[]
    for path in sorted((root/'cells').glob('*/metrics.json')):
        r=json.loads(path.read_text());r['path']=path.parent;rs.append(r)
    if len(rs)!=16:raise RuntimeError(f'Need16 targets; found{len(rs)}')
    def group(capacity=None,regime=None,corpus=None):
        return [r for r in rs if (capacity is None or r['capacity']==capacity) and (regime is None or r['regime']==regime) and (corpus is None or r['corpus']==corpus)]
    def mean_metric(g,m,k='auc'):return float(np.mean([r['metrics'][m][k] for r in g]))
    def finish(fig,name):
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    macro={m:{k:mean_metric(rs,m,k) for k in ['auc','tpr_at_1pct_fpr','average_precision']} for m in METHODS}
    strata={f'{cap}/{regime}':{m:{k:mean_metric(group(cap,regime),m,k) for k in ['auc','tpr_at_1pct_fpr','average_precision']} for m in METHODS}
        for cap in ['small','larger'] for regime in ['standard','repeated']}
    (root/'macro_metrics.json').write_text(json.dumps(dict(overall=macro,strata=strata),indent=2)+'\n')
    exposure={cap:{m:{k:float(np.mean([r['high_exposure_metrics'][m][k] for r in group(cap,'repeated')]))
                     for k in ['auc','average_precision']} for m in METHODS if m not in PRIVILEGED}
              for cap in ['small','larger']}
    (root/'high_exposure_metrics.json').write_text(json.dumps(exposure,indent=2)+'\n')
    moderate={}
    for cap in ['small','larger']:
        cases=[]
        for r in group(cap,'repeated'):
            with np.load(r['path']/'scores.npz') as z:
                mask=z['high_exposure']==0
                from spw_methods import evaluate_scores as evaluate_subgroup
                cases.append({m:evaluate_subgroup(z['labels'][mask],z[m][mask])['auc'] for m in [PRIMARY,'Membrain replicate16','Membrain pooled32','Greedy whole-suffix']})
        moderate[cap]={m:float(np.mean([c[m] for c in cases])) for m in cases[0]}
    (root/'moderate_exposure_diagnostic.json').write_text(json.dumps(dict(exploratory=True,
       definition='Repeated-condition12-presentation members vs nonmembers, excluding96-presentation items from evaluation only',metrics=moderate),indent=2)+'\n')
    groups={}
    for cap in ['small','larger']:
        for regime in ['standard','repeated']:
            subset=group(cap,regime);summary={}
            for name in [PRIMARY,'Membrain replicate16','Membrain pooled32']:
                selected=[r['clustering'][name] for r in subset if r['clustering'][name]['selected_two']]
                summary[name]=dict(targets=len(subset),selected_two=len(selected))
                for label in ['membership','high_exposure']:
                    values=[v[label] for v in selected if v[label]]
                    summary[name][label]={k:float(np.mean([v[k] for v in values])) for k in ['precision','recall','balanced_accuracy','false_positive_rate']} if values else None
            groups[f'{cap}/{regime}']=summary
    (root/'group_summary.json').write_text(json.dumps(groups,indent=2)+'\n')

    comparison=[PRIMARY,'Membrain pooled32','Membrain nested3','Greedy whole-suffix','LOSS','Min-k20%']
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
    for ax,(cap,regime) in zip(axes.flat,[(c,e) for c in ['small','larger'] for e in ['standard','repeated']]):
        subset=group(cap,regime)
        for i,m in enumerate(comparison):
            values=np.array([r['metrics'][m]['auc'] for r in subset]);v=values.mean()
            ax.bar(i,v,color='#ad315f' if m==PRIMARY else '#b28a58' if m in PRIVILEGED else '#518ba5')
            ax.errorbar(i,v,yerr=np.array([[v-values.min()],[values.max()-v]]),fmt='none',color='black',capsize=3)
        labels=[m+' *' if m in PRIVILEGED else m for m in comparison]
        ax.set_xticks(range(len(labels)),labels,rotation=30,ha='right');ax.set_ylim(0,1.02);ax.axhline(.5,color='#999999',ls='--')
        ax.set_title(f'{cap}: {regime}');ax.set_ylabel('Membership ROC AUC')
    fig.suptitle('Fresh validation · means/ranges across corpora and pools · *privileged references')
    fig.tight_layout();finish(fig,'comparison')

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,(corpus,regime) in zip(axes.flat,[(c,e) for c in ['ag_news','wikitext2'] for e in ['standard','repeated']]):
        for cap in ['small','larger']:
            subset=group(cap,regime,corpus);values=[mean_metric(subset,m) for m in ['Membrain nested3',PRIMARY,'Membrain pooled32']]
            ax.plot([3,16,32],values,'-o',label=cap)
        ax.axhline(.5,color='#999999',ls='--');ax.set_ylim(0,1.02);ax.set_title(f'{corpus}: {regime}')
        ax.set_xlabel('Stochastic completions per prefix');ax.set_ylabel('Membership AUC')
    axes[0,0].legend();fig.suptitle('Fixed-budget controls: first3 nested in primary16;32 pools two independent batches')
    fig.tight_layout();finish(fig,'query_budget')

    fig,ax=plt.subplots(figsize=(7,6))
    for cap,color in [('small','#4d8ba5'),('larger','#ae345e')]:
        for regime,marker in [('standard','o'),('repeated','s')]:
            subset=group(cap,regime);ax.scatter([r['metrics'][PRIMARY]['auc'] for r in subset],
                [r['metrics']['Membrain replicate16']['auc'] for r in subset],color=color,marker=marker,label=f'{cap}/{regime}')
    ax.plot([.4,1],[.4,1],'--',color='#888888');ax.set_xlabel('Primary16 membership AUC');ax.set_ylabel('Independent16 replicate AUC');ax.legend(fontsize=8)
    fig.tight_layout();finish(fig,'query_replicates')

    fig,ax=plt.subplots(figsize=(7,6));abstentions=0
    for cap,color in [('small','#4d8ba5'),('larger','#ae345e')]:
        for regime,marker in [('standard','o'),('repeated','s')]:
            values=[]
            for r in group(cap,regime):
                m=r['clustering'][PRIMARY]['membership']
                if m:values.append(m)
                else:abstentions+=1
            if values:ax.scatter([m['recall'] for m in values],[m['precision'] for m in values],color=color,marker=marker,label=f'{cap}/{regime}')
    ax.set_xlim(0,1.02);ax.set_ylim(0,1.02);ax.set_xlabel('Held-out membership recall');ax.set_ylabel('Held-out membership precision')
    ax.set_title(f'Unsupervised high-recovery groups · {abstentions}/16 abstentions');ax.legend(fontsize=8)
    fig.tight_layout();finish(fig,'heldout_groups')

    fig,ax=plt.subplots(figsize=(11,8));d=[];lo=[];hi=[];names=[]
    for r in rs:
        delta=r['metrics'][PRIMARY]['differences']['Greedy whole-suffix'];v=delta['delta'];d.append(v)
        lo.append(v-delta['ci95'][0]);hi.append(delta['ci95'][1]-v);names.append(f'{r["capacity"]}/{r["corpus"]}/{r["seed"]}/{r["regime"]}')
    ax.errorbar(d,range(len(rs)),xerr=[lo,hi],fmt='o',capsize=3,color='#ad315f');ax.axvline(0,ls='--',color='#888888')
    ax.set_yticks(range(len(rs)),names);ax.invert_yaxis();ax.set_xlabel('Primary16 AUC minus greedy whole-suffix · paired sample-bootstrap95%CI')
    fig.tight_layout();finish(fig,'paired_gain')

    # Saturation is a reporting diagnostic, never a scoring/model-selection input.
    saturation=[];curve_data={}
    for r in rs:
        with (r['path']/'primary_transcripts.jsonl').open() as f:records=[json.loads(line) for line in f]
        evaluated=[score_record(record)[1] for record in records]
        with np.load(r['path']/'scores.npz') as z:y=z['labels'];high=z['high_exposure'];primary=z[PRIMARY]
        floor=np.log(.5/17);zero=np.isclose(primary,floor,rtol=0,atol=1e-12)
        saturation.append(dict(capacity=r['capacity'],corpus=r['corpus'],seed=r['seed'],regime=r['regime'],
          no_hits_in_any_probe_member=float(zero[y==1].mean()),no_hits_in_any_probe_nonmember=float(zero[y==0].mean())))
        curve_data[r['path'].name]=(records,evaluated,y,high)
    (root/'sampling_saturation.json').write_text(json.dumps(saturation,indent=2)+'\n')
    grid=np.linspace(.125,1,8)
    for cap in ['small','larger']:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
        for ax,(corpus,regime) in zip(axes.flat,[(c,e) for c in ['ag_news','wikitext2'] for e in ['standard','repeated']]):
            data={'ordinary members':[],'nonmembers':[]}
            if regime=='repeated':data['heavily repeated members']=[]
            for r in group(cap,regime,corpus):
                records,curves,y,high=curve_data[r['path'].name]
                values=np.array([np.interp(grid,np.array(c['progress'])/(1-4/len(words(t['text']))),c['frequencies']) for t,c in zip(records,curves)])
                data['ordinary members'].extend(values[(y==1)&(high==0)]);data['nonmembers'].extend(values[y==0])
                if regime=='repeated':data['heavily repeated members'].extend(values[high==1])
            for name,values in data.items():ax.plot(grid,np.mean(values,0),marker='.',label=name)
            ax.legend(fontsize=7);ax.set_ylim(0,1.02);ax.set_title(f'{corpus}: {regime}')
            ax.set_xlabel('Fraction of available prefix');ax.set_ylabel('Correct next-word sampling frequency')
        fig.suptitle(f'{cap} capacity · fixed16 text-only samples per prefix')
        fig.tight_layout();finish(fig,f'recovery_curves_{cap}')

    rows=[]
    for r in rs:
        for m,v in r['metrics'].items():
            delta=v['differences']['Greedy whole-suffix'];rows.append(dict(capacity=r['capacity'],corpus=r['corpus'],seed=r['seed'],pool_id=r['pool_id'],regime=r['regime'],method=m,
              access='probability reference' if m in PRIVILEGED else 'text only',auc=v['auc'],auc_ci_low=v['auc_ci95'][0],auc_ci_high=v['auc_ci95'][1],
              tpr_at_1pct_fpr=v['tpr_at_1pct_fpr'],average_precision=v['average_precision'],delta_vs_greedy_whole=delta['delta'],delta_ci_low=delta['ci95'][0],delta_ci_high=delta['ci95'][1]))
    with (root/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Membrain next-word recovery: prospective fresh-data validation','',
      '**Same fixed score; fresh sentences, fresh target models, two capacities, independent query replication.** '
      'The attacker sees completion strings only. No membership labels enter score/model selection, unlabeled mixture fitting, component orientation, or calibration. '
      'Target-training/evaluation labels are used by the benchmark owner, separately from attack inference.',
      '', '## Protocol and scope','',
      'Primary16 is the EXACT development score: up to eight increasing word prefixes, 16 IID temperature 0.8 short completions '
      '(cap 8 new tokens), count correct first lexical words, Jeffreys smoothing(h+.5)/17, and early-weighted log-frequency area. '
      'The eight-stage grid is deduplicated when word counts coincide. The method estimates a text-event probability; it never reads '
      'the model’s true token probabilities. The primary score and generation settings were not changed in this study.',
      '', 'Two new 384-sentence pools per corpus: AG News train-source descriptions and WikiText-2 validation-source paragraphs, '
      'excluding all previous exact document chunks/full reference sentences and tokenizer background. 16 targets = two capacities '
      '× two corpora × two exposure regimes × two seeds. Seeds 9101/9102 use separate pools. Capacities/regimes share a seed’s pool '
      'and membership assignment; these are not 16 independent datasets. Each target evaluates 128 members/128 nonmembers, '
      'blindly shuffled before queries. Small 182,016 parameters; larger 458,592. Both are tiny local models, not commercial LLMs.',
      '', 'An independent 16-sample query repeat measures stability. 32-sample pooling and a nested 3-sample subset were predeclared '
      'secondary budget controls. Greedy whole-suffix generation provides an output-only baseline. LOSS/Min-k/SPW are privileged '
      'owner-side references computed only after scores and unlabeled mixtures are saved. No reference enters the detector.',
      '', '## Primary comparison','',
      '| Method | Access | Macro membership AUC | TPR at ≤1%FPR | AP |','|---|---|---:|---:|---:|']
    for m,v in macro.items():lines.append(f'| {m} | {"Privileged probabilities" if m in PRIVILEGED else "Completion text only"} | {v["auc"]:.4f} | {100*v["tpr_at_1pct_fpr"]:.2f}% | {v["average_precision"]:.4f} |')
    lines+=['',f'Primary16 minus greedy whole-suffix macro AUC: **{macro[PRIMARY]["auc"]-macro["Greedy whole-suffix"]["auc"]:+.4f}**. '
      'Development AUC was 0.7106 on previously evaluated data; compare it descriptively, not as a paired baseline. '
      'Fresh source partitions, capacities and pool structure changed. No best secondary result replaces the primary.',
      '', '| Capacity / exposure | Primary16 | Replicate16 | Pooled32 | Nested3 | Greedy whole | Min-k20% |','|---|---:|---:|---:|---:|---:|---:|']
    for key,sub in strata.items():lines.append(f'| {key} | '+' | '.join(f'{sub[m]["auc"]:.4f}' for m in [PRIMARY,'Membrain replicate16','Membrain pooled32','Membrain nested3','Greedy whole-suffix','Min-k20%'])+' |')
    lines+=['','## Heavy exposure as a separate target','',
      'Repeated conditions expose32 member sentences96 times each and other members12 times. This table ranks that '
      'known high-exposure group against all other candidates; it is not an oracle label for an internal memory state. '
      'Strong recovery and ordinary training membership are different detection targets.',
      '', '| Text-only score | Small: high-exposure AUC | Larger: high-exposure AUC |','|---|---:|---:|']
    for m in [PRIMARY,'Membrain replicate16','Membrain pooled32','Greedy next-word','Greedy whole-suffix']:
        lines.append(f'| {m} | {exposure["small"][m]["auc"]:.4f} | {exposure["larger"][m]["auc"]:.4f} |')
    lines+=['','### Exploratory subgroup check: remove the heaviest items','',
       'This descriptive check was added after the primary results. Exclude the96-presentation group from evaluation, '
       'leaving96 members seen12 times and128 nonmembers in each repeated target. Models are NOT retrained and no '
       'score/threshold is changed. This asks whether the ranking signal is entirely driven by the most repeated items; '
       'it is not a newly validated primary or an isolated causal estimate of repetition.',
       '', '| Capacity | Primary16 AUC | Greedy whole-suffix AUC |','|---|---:|---:|']
    for cap,v in moderate.items():lines.append(f'| {cap} | {v[PRIMARY]:.4f} | {v["Greedy whole-suffix"]:.4f} |')
    lines+=['','## Every primary target and paired uncertainty','',
      '1000 paired stratified sample-bootstrap draws per target, conditional on fixed target/queries. Intervals omit training '
      'and generation-sampling uncertainty and multiple-comparison correction; query replication is reported separately.',
      '', '| Target | Primary16 AUC | Δ vs greedy whole | Paired 95% CI | Replicate AUC |','|---|---:|---:|---|---:|']
    for r in rs:
        v=r['metrics'][PRIMARY];d=v['differences']['Greedy whole-suffix']
        lines.append(f'| {r["capacity"]}/{r["corpus"]}/{r["seed"]}/{r["regime"]} | {v["auc"]:.4f} | {d["delta"]:+.4f} | [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}] | {r["metrics"]["Membrain replicate16"]["auc"]:.4f} |')
    lines+=['','## Query noise and unsupervised held-out groups','',
      'Mixtures fit 128 unlabeled discovery scores, choose one vs two Gaussian components by BIC, orient by higher recovery, '
      'and evaluate 128 other sentences. They abstain if one component is preferred. No class is reoriented using labels. '
      'Component posterior is not calibrated membership probability. Precision depends on the held-out prevalence shown below.',
      '', '| Target | Test members/nonmembers | Primary groups | Membership precision | Recall | Balanced accuracy | Query rank correlation | High-group Jaccard |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return '—' if v is None else f'{v:.3f}'
    for r in rs:
        c=r['clustering'][PRIMARY];m=c['membership'];s=r['stability']
        lines.append(f'| {r["capacity"]}/{r["corpus"]}/{r["seed"]}/{r["regime"]} | {c["heldout_members"]}/{c["heldout_nonmembers"]} | {2 if c["selected_two"] else 1} | '
          f'{fmt(m["precision"] if m else None)} | {fmt(m["recall"] if m else None)} | {fmt(m["balanced_accuracy"] if m else None)} | {fmt(s["spearman"])} | {fmt(s["high_group_jaccard"])} |')
    lines+=['','Full primary/replicate/pooled group metrics are in `group_summary.json`. Gaussian mixtures on bounded/censored '
      'scores are heuristics; two components are not proof of member/nonmember classes. Weakly learned members and nonmembers '
      'may share low-recovery groups. Query noise can change grouping even when aggregate AUC is similar.',
      '', '| Capacity / exposure | Query rank correlation | High-group Jaccard | No hits at any prefix: members | No hits: nonmembers |',
      '|---|---:|---:|---:|---:|']
    for cap in ['small','larger']:
        for regime in ['standard','repeated']:
            subset=group(cap,regime);sat=[v for v in saturation if v['capacity']==cap and v['regime']==regime]
            corr=[r['stability']['spearman'] for r in subset if r['stability']['spearman'] is not None]
            jac=[r['stability']['high_group_jaccard'] for r in subset if r['stability']['high_group_jaccard'] is not None]
            lines.append(f'| {cap}/{regime} | {fmt(float(np.mean(corr)) if corr else None)} | {fmt(float(np.mean(jac)) if jac else None)} | '
               f'{100*np.mean([v["no_hits_in_any_probe_member"] for v in sat]):.1f}% | {100*np.mean([v["no_hits_in_any_probe_nonmember"] for v in sat]):.1f}% |')
    lines+=[
      '', '## Interpretation and limits','',
      'The right validation question is whether ordinary training membership improves reliably, not merely whether '
      'heavily repeated sentences recover. Repeated exposure also changes optimization budget. Heavy exposure is known '
      'training frequency, not an oracle memory label. All conditions/capacities and controls are reported, including failures.',
      '', 'Short-generation caps can censor first words; rare correct words may never appear in 16 samples. '
      '`sampling_saturation.json` is a reporting diagnostic of candidates with zero matches at every probe, not a scoring '
      'or selection input. More samples change query costs and the smoothing floor; compare rankings within methods, '
      'not raw scores across budgets. Candidate word-count control checks a non-model metadata signal.',
      '', 'No large-model, external-domain or commercial-API performance claim follows. Exact exclusions do not remove '
      'semantic overlap; sentence segmentation/WikiText artifacts remain. At 128 nonmembers, empirical 1%FPR allows '
      'one false positive. No deployment threshold or calibrated membership probability is fitted. Prior research motivated '
      'the unchanged hypothesis; this is prospective internal confirmation, not externally preregistered replication.',
      '', f'Total completions including query repeats/greedy controls: **{sum(r["query_stats"]["completions"] for r in rs):,}**; '
      f'backend generated tokens: **{sum(r["query_stats"]["generated_tokens"] for r in rs):,}**. '
      'Public prefix lengths are not suffix information; the API never receives the reference continuation. '
      'Raw transcripts and saved mixtures allow string-only replay without model weights or a model tokenizer.',
      '', 'See [frozen protocol](../../experiments/MEMBRAIN_TEXT_VALIDATION.md).','', '## Plots','']
    for name in ['comparison','query_budget','query_replicates','heldout_groups','paired_gain','recovery_curves_small','recovery_curves_larger']:
        lines.extend([f'![{name}](plots/{name}.png)',''])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(overall=macro,groups=groups),indent=2))


if __name__=='__main__':main()
