"""Output-only recovery, unlabeled grouping, and stronger-access references."""
import argparse
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from membrain_text import words

PRIMARY='Membrain text'
REFERENCES=['LOSS','Min-k20%','SPW-MIA']


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('results/membrain_text'));root=p.parse_args().input
    out=root/'plots';out.mkdir(exist_ok=True);rs=[]
    for path in sorted((root/'cells').glob('*/metrics.json')):
        r=json.loads(path.read_text());r['path']=path.parent;rs.append(r)
    if len(rs)!=12:raise RuntimeError(f'Need12 complete targets; found{len(rs)}')
    scenarios=[(c,e) for c in ['ag_news','wikitext2'] for e in ['standard','repeated']]
    def group(c,e):return [r for r in rs if r['corpus']==c and r['regime']==e]
    def finish(fig,name):
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    methods=list(rs[0]['metrics']);macro={m:{k:float(np.mean([r['metrics'][m][k] for r in rs]))
        for k in ['auc','tpr_at_1pct_fpr','average_precision']} for m in methods}
    exposure={m:{k:float(np.mean([r['high_exposure_metrics'][m][k] for r in rs if r['regime']=='repeated']))
        for k in ['auc','average_precision']} for m in methods if m not in REFERENCES}
    (root/'macro_metrics.json').write_text(json.dumps(dict(membership=macro,high_exposure=exposure),indent=2)+'\n')
    group_summary={}
    for regime in ['standard','repeated']:
        chosen=[r for r in rs if r['regime']==regime and r['clustering']['selected_two']]
        summary=dict(targets=6,two_components_selected=len(chosen))
        for label in ['membership','high_exposure']:
            values=[r['clustering'][label] for r in chosen if r['clustering'][label]]
            summary[label]={k:float(np.mean([v[k] for v in values])) for k in ['precision','recall','balanced_accuracy','false_positive_rate']} if values else None
        group_summary[regime]=summary
    (root/'unsupervised_group_summary.json').write_text(json.dumps(group_summary,indent=2)+'\n')
    probe=[]
    for r in rs:
        path=r['path']/'word_probe_metrics.json'
        if path.exists():probe.append(json.loads(path.read_text()))
    if probe and len(probe)!=12:raise RuntimeError('Finish all12 exploratory word probes before the combined report')
    probe_macro=None
    if probe:
        probe_macro={m:{k:float(np.mean([p['metrics'][m][k] for p in probe])) for k in ['auc','tpr_at_1pct_fpr','average_precision']} for m in probe[0]['metrics']}
        (root/'word_probe_macro.json').write_text(json.dumps(probe_macro,indent=2)+'\n')
        probe_group_summary={}
        for regime in ['standard','repeated']:
            values=[p['clustering']['membership'] for r,p in zip(rs,probe) if r['regime']==regime and p['clustering']['membership']]
            probe_group_summary[regime]={k:float(np.mean([v[k] for v in values])) for k in ['precision','recall','balanced_accuracy','false_positive_rate']} if values else None
        (root/'word_probe_group_summary.json').write_text(json.dumps(probe_group_summary,indent=2)+'\n')
        # Exploratory budget diagnostic: SAME new word scorer on original3 samples.
        from membrain_word_probe import score_record as score_words
        from spw_methods import evaluate_scores
        original_word_metrics=[]
        for r in rs:
            with (r['path']/'transcripts.jsonl').open() as f:records=[json.loads(line) for line in f]
            scores3=np.array([score_words(record)[0]['Membrain next-word probe'] for record in records])
            with np.load(r['path']/'scores.npz') as z:labels=z['labels']
            original_word_metrics.append(evaluate_scores(labels,scores3))
        budget_diagnostic={k:float(np.mean([m[k] for m in original_word_metrics])) for k in ['auc','tpr_at_1pct_fpr','average_precision']}
        (root/'word_probe_original_sample_budget.json').write_text(json.dumps(dict(exploratory=True,description='Word scorer replayed on original3 full-completion samples; no new queries',metrics=budget_diagnostic),indent=2)+'\n')
    compare=[PRIMARY,'Membrain prefix advantage','Sampled whole-suffix edit','Greedy whole-suffix edit','Single longest-prefix recovery','Min-k20%']
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharey=True)
    for ax,(c,e) in zip(axes.flat,scenarios):
        for i,m in enumerate(compare):
            vals=np.array([r['metrics'][m]['auc'] for r in group(c,e)]);v=vals.mean()
            ax.bar(i,v,color='#b68a57' if m in REFERENCES else '#a93460' if m==PRIMARY else '#4c8fa8')
            ax.errorbar(i,v,yerr=np.array([[v-vals.min()],[vals.max()-v]]),fmt='none',color='black',capsize=3)
        ax.set_xticks(range(len(compare)),[m+' *' if m in REFERENCES else m for m in compare],rotation=35,ha='right')
        ax.set_ylim(0,1.02);ax.axhline(.5,color='#999999',ls='--');ax.set_title(f'{c}: {e}');ax.set_ylabel('Membership ROC AUC')
    fig.suptitle('Completion-text scores · *privileged probability reference · seed means/ranges')
    fig.tight_layout();finish(fig,'membership_comparison')
    if probe:
        fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
        for ax,(c,e) in zip(axes.flat,scenarios):
            indices=[i for i,r in enumerate(rs) if r['corpus']==c and r['regime']==e]
            names=['Fixed-tail recovery','Greedy whole suffix','Next-word probe','Greedy next word','LOSS *','Min-k20% *']
            values=[np.mean([rs[i]['metrics'][PRIMARY]['auc'] for i in indices]),
              np.mean([rs[i]['metrics']['Greedy whole-suffix edit']['auc'] for i in indices]),
              np.mean([probe[i]['metrics']['Membrain next-word probe']['auc'] for i in indices]),
              np.mean([probe[i]['metrics']['Greedy next-word recovery']['auc'] for i in indices]),
              np.mean([rs[i]['metrics']['LOSS']['auc'] for i in indices]),np.mean([rs[i]['metrics']['Min-k20%']['auc'] for i in indices])]
            bars=ax.bar(range(6),values,color=['#4c8fa8','#4c8fa8','#a93460','#a93460','#b68a57','#b68a57'])
            bars[2].set_hatch('//');bars[3].set_hatch('//');ax.set_xticks(range(6),names,rotation=30,ha='right')
            ax.axhline(.5,color='#999999',ls='--');ax.set_ylim(0,1.02);ax.set_title(f'{c}: {e}');ax.set_ylabel('Membership ROC AUC')
        fig.suptitle('Striped: exploratory short-completion control · *privileged reference')
        fig.tight_layout();finish(fig,'word_probe_comparison')

    grid=np.linspace(.125,1,8);pools=json.loads((root/'sentences.json').read_text())
    loaded={}
    for r in rs:
        curves=json.loads((r['path']/'recovery_curves.json').read_text())
        with np.load(r['path']/'scores.npz') as z:ids=z['eval_ids']
        lengths=[len(words(pools[r['corpus']][i])) for i in ids]
        loaded[r['path'].name]=(curves,lengths)
    for key,label in [('tail_word','Fixed four-word recovery similarity'),('full_word','Whole-suffix recovery similarity'),
                       ('greedy_full','Greedy whole-suffix word-edit similarity')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
        for ax,(c,e) in zip(axes.flat,scenarios):
            values={'ordinary members':[],'nonmembers':[]}
            if e=='repeated':values['heavily repeated members']=[]
            for r in group(c,e):
                curves,lengths=loaded[r['path'].name]
                v=np.array([np.interp(grid,np.array(q['progress'])/(1-4/w),q['curves'][key]) for q,w in zip(curves,lengths)])
                values['ordinary members'].extend(v[32:128] if e=='repeated' else v[:128]);values['nonmembers'].extend(v[128:])
                if e=='repeated':values['heavily repeated members'].extend(v[:32])
            for name,vs in values.items():ax.plot(grid,np.mean(vs,0),marker='.',label=name)
            ax.set_ylim(0,1.02);ax.legend(fontsize=7);ax.set_title(f'{c}: {e}')
            ax.set_xlabel('Fraction of available prefix (final four words withheld)');ax.set_ylabel(label)
        fig.tight_layout();finish(fig,'curves_'+key)

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,(c,e) in zip(axes.flat,scenarios):
        r=next(r for r in group(c,e) if r['seed']==8101)
        mix=json.loads((r['path']/'unsupervised_mixture.json').read_text());m=mix['model'];idx=np.array(mix['heldout'])
        with np.load(r['path']/'scores.npz') as z:y=z['labels'][idx];s=z[PRIMARY][idx]
        bins=np.linspace(0,1,16)
        ax.hist(s[y==0],bins=bins,histtype='step',density=True,lw=2,label='Nonmembers: evaluation labels')
        ax.hist(s[y==1],bins=bins,histtype='step',density=True,lw=2,label='Members: evaluation labels')
        x=np.linspace(0,1,401)
        if m['selected_two']:
            for j in range(2):
                mu,var,w=m['means'][j],m['variances'][j],m['weights'][j]
                ax.plot(x,w*np.exp(-.5*(x-mu)**2/var)/np.sqrt(2*np.pi*var),'--',color='#777777',label='Unlabeled fitted component' if j==0 else None)
        ax.set_title(f'{c}: {e}; BIC selects {2 if m["selected_two"] else 1} components')
        ax.set_xlabel('Membrain text recovery score');ax.set_ylabel('Density on held-out texts')
    axes[0,0].legend(fontsize=7);fig.suptitle('Discovery fit uses NO labels; membership labels shown only to evaluate')
    fig.tight_layout();finish(fig,'unsupervised_groups')

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for row,c in enumerate(['ag_news','wikitext2']):
        for col,stage in enumerate([.2,.8]):
            ax=axes[row,col];sets={'ordinary members':[],'nonmembers':[],'heavily repeated members':[]}
            for r in group(c,'repeated'):
                curves,lengths=loaded[r['path'].name]
                for i,(q,w) in enumerate(zip(curves,lengths)):
                    relative=np.array(q['progress'])/(1-4/w);j=int(np.abs(relative-stage).argmin())
                    label='heavily repeated members' if i<32 else 'ordinary members' if i<128 else 'nonmembers'
                    sets[label].extend(q['word_distance_samples'][j])
            for label,vs in sets.items():
                a=np.sort(vs);ax.step(np.r_[0,a,1],np.r_[0,np.arange(1,len(a)+1)/len(a),1],where='post',label=label)
            ax.set_title(f'{c}, prefix near {stage:.0%} of available context');ax.set_xlabel('Normalized word-edit distance (lower = closer)');ax.set_ylabel('Empirical CDF')
    axes[0,0].legend(fontsize=7);fig.tight_layout();finish(fig,'distance_distributions')

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for row,c in enumerate(['ag_news','wikitext2']):
        for col,stage in enumerate([.2,.8]):
            ax=axes[row,col];sets={'ordinary members':[],'nonmembers':[],'heavily repeated members':[]}
            for r in group(c,'repeated'):
                curves,lengths=loaded[r['path'].name]
                for i,(q,w) in enumerate(zip(curves,lengths)):
                    relative=np.array(q['progress'])/(1-4/w);j=int(np.abs(relative-stage).argmin())
                    label='heavily repeated members' if i<32 else 'ordinary members' if i<128 else 'nonmembers'
                    sets[label].append(1-q['curves']['greedy_full'][j])
            for label,vs in sets.items():
                a=np.sort(vs);ax.step(np.r_[0,a,1],np.r_[0,np.arange(1,len(a)+1)/len(a),1],where='post',label=label)
            ax.set_title(f'{c}, prefix near {stage:.0%} of available context');ax.set_xlabel('Greedy whole-suffix normalized word-edit distance');ax.set_ylabel('Empirical CDF across sentences')
    axes[0,0].legend(fontsize=7);fig.tight_layout();finish(fig,'greedy_distance_distributions')

    rows=[]
    for r in rs:
        for m,v in r['metrics'].items():
            rows.append(dict(corpus=r['corpus'],regime=r['regime'],seed=r['seed'],method=m,
                access='privileged probabilities' if m in REFERENCES else 'completion text only',
                auc=v['auc'],auc_ci_low=v['auc_ci95'][0],auc_ci_high=v['auc_ci95'][1],
                tpr_at_1pct_fpr=v['tpr_at_1pct_fpr'],average_precision=v['average_precision']))
    with (root/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    repeated=group_summary['repeated']['membership']
    grouping_finding=(f'In repeated conditions the unsupervised held-out high-recovery group averages '
      f'**{100*repeated["precision"]:.1f}% membership precision** and '
      f'**{100*repeated["recall"]:.1f}% recall**. ' if repeated else
      'All repeated-condition mixtures abstained from two-class assignment. ')
    lines=['# Membrain: output-only recovery distributions and unsupervised grouping','',
      '**The detector uses completion TEXT only. No hidden states, target probabilities, token IDs, gradients, or weights enter its score or mixture.** '
      'An opaque local service emulates a text-only API; no commercial hosted model was queried. '
      'All mixtures fit unlabeled discovery scores; no membership-supervised fitting/calibration or label-based component orientation occurs.','',
      '## Main findings','',
      f'The predefined fixed-tail score achieves membership AUC **{macro[PRIMARY]["auc"]:.4f}**; '
      f'the predefined greedy whole-suffix alternative achieves **{macro["Greedy whole-suffix edit"]["auc"]:.4f}**. '
      f'Greedy whole-suffix edit ranks the heavily exposed group with AUC **{exposure["Greedy whole-suffix edit"]["auc"]:.4f}** '
      'in the six repeated-exposure targets. These are different evaluation targets: strong recovery is much easier to detect than all training membership.',
      '',grouping_finding+
      'Many ordinary members remain in the low-recovery group. Two components are also selected in ordinary training conditions, '
      'where membership grouping is near chance; finding two components alone does not establish member/nonmember classes.',
      '',
      '## Procedure','',
      'Supply increasing word prefixes of complete sentences, withhold at least four words, and request three IID temperature 0.8 completions '
      'plus a separate greedy completion at each stage. Compare outputs against original suffixes using normalized word edit, character edit, '
      'ROUGE-L and exact match. Keep a separate empirical distribution per prefix; different prefix conditions are not IID samples. '
      'Three draws per condition give only coarse distributions.',
      '', 'Primary **Membrain text** integrates expected fixed-four-word edit similarity over prefix lengths, favoring earlier recovery. '
      'Equivalently it uses Wasserstein-1 distance of the empirical scalar distance distribution to perfect zero-distance recovery. '
      'A reversed-prefix control and whole-suffix, greedy, quantile and recovery-speed alternatives were fixed in advance and are all reported.',
      '', 'Fit one- versus two-Gaussian mixtures on 128 unlabeled discovery sentences by BIC; evaluate on 128 different sentences. '
      'The higher-mean component is always called high recovery. If one component is preferred, abstain from two-class assignment. '
      'A component posterior is not a membership probability, and two recovery groups need not be member/nonmember groups.',
      '', '## Benchmark and comparison','',
      'Twelve new sentence-trained tiny models: two corpora × two exposure regimes × three seeds. 384 fresh documents/sentences per corpus '
      'exclude previous exact document chunks and tokenizer background. Each target evaluates 128 members and 128 nonmembers. '
      'Repeated conditions expose 32 of the evaluated member sentences 96 times, with other members 12 times; ordinary training exposes '
      'all members 4 times. Heavy exposure is not an oracle memorization label. Sentence training includes newline termination, ignored '
      'padding, and variable lengths. This changes the benchmark; previous studies’ numbers are not paired comparisons.',
      '', '| Method | Access | Macro membership AUC | TPR at ≤1% FPR | AP |','|---|---|---:|---:|---:|']
    for m,v in macro.items():lines.append(f'| {m} | {"Privileged probabilities" if m in REFERENCES else "Completion text only"} | {v["auc"]:.4f} | {100*v["tpr_at_1pct_fpr"]:.2f}% | {v["average_precision"]:.4f} |')
    lines+=['','LOSS, Min-k and SPW are stronger-access owner-side references computed only AFTER output-only scores and unlabeled mixtures '
      'were saved. They are not callable by the black-box detector. Output-only alternatives have different query budgets: sampled methods '
      'use three draws per prefix; greedy baselines use one. Reversed-prefix control adds another set of queries. All settings are fixed.',
      '', '## Distinguishing heavy exposure','',
      'This secondary evaluation compares the known 96-presentation group against all other candidates in repeated-exposure targets. '
      'It evaluates training frequency, not proof of an internal memory state.','',
      '| Output-only method | Heavy-exposure ROC AUC | AP |','|---|---:|---:|']
    for m,v in exposure.items():lines.append(f'| {m} | {v["auc"]:.4f} | {v["average_precision"]:.4f} |')
    lines+=['','## Unsupervised grouping on held-out sentences','',
      'Membership and heavy-exposure evaluation labels are used ONLY below to assess the groups. No component is reoriented to maximize accuracy. '
      '“Abstain” means BIC favored one component. Values refer to the 128 held-out sentences, not discovery fitting.',
      '', '| Target | BIC components | Membership precision | Membership recall | Membership balanced accuracy | Heavy-exposure balanced accuracy |',
      '|---|---:|---:|---:|---:|---:|']
    def fmt(x):return '—' if x is None else f'{x:.3f}'
    for r in rs:
        c=r['clustering'];m=c['membership'];h=c['high_exposure']
        lines.append(f'| {r["corpus"]}/{r["regime"]}/{r["seed"]} | {2 if c["selected_two"] else 1} | '
            f'{fmt(m["precision"] if m else None)} | {fmt(m["recall"] if m else None)} | {fmt(m["balanced_accuracy"] if m else None)} | '
            f'{fmt(h["balanced_accuracy"] if h else None)} |')
    selected=sum(r['clustering']['selected_two'] for r in rs)
    first_mix=json.loads((rs[0]['path']/'unsupervised_mixture.json').read_text())
    with np.load(rs[0]['path']/'scores.npz') as z:
        heldout_labels=z['labels'][np.array(first_mix['heldout'])]
    heldout_members=int(heldout_labels.sum());heldout_nonmembers=len(heldout_labels)-heldout_members
    lines+=['',f'Two components selected in **{selected}/12** targets. '
      'Gaussian mixtures are heuristics for bounded/discrete recovery scores; BIC is not proof of two natural membership classes. '
      'Weakly learned members can share the low-recovery distribution with nonmembers. High recovery can also arise from general '
      'language predictability. Interpret high-recovery groups as memorization-like hypotheses, not certain membership decisions.',
      '',f'The fixed random held-out index split contains {heldout_members} members and {heldout_nonmembers} nonmembers per target '
      '(these counts are not used in fitting). Precision depends on this evaluation prevalence; it is not an external positive predictive value.',
      '', '## Query budget and limits','',
      f'Total emitted completions across true/reversed prefixes and all targets: **{sum(r["query_stats"]["completions"] for r in rs):,}**. '
      f'Total backend-generated tokens (including termination): **{sum(r["query_stats"]["generated_tokens"] for r in rs):,}**. '
      'These are local research costs, not a cost estimate for a hosted API. Generation may terminate early or hit the fixed cap; '
      'last-four-word comparison may therefore compare a different ending when the completion does not finish the intended sentence.',
      '', 'Small 182,016-parameter models, 12–40-word sentences, shared corpus pools across three seeds. No large-model or commercial API '
      'performance follows. Exact deduplication does not rule out semantic overlap. Simple sentence segmentation and WikiText artifacts '
      'are limitations. Repeated training changes both duplication and optimization budget. At 128 nonmembers, empirical 1%FPR permits '
      'one false positive; no deployment FPR guarantee or membership probability is calibrated. Sample-bootstrap intervals condition on '
      'fitted targets and fixed sampled completions; they omit training/generation-sampling uncertainty and multiplicity correction.',
      '', 'The curve displays interpolate prefix grids only for visualization; scores use each sentence’s actual word-prefix fractions. '
      'Raw transcripts contain no benchmark labels and can be rescored without PyTorch, model weights, or a model tokenizer. '
      'See [protocol](../../experiments/MEMBRAIN_TEXT_PROTOCOL.md) and `score_membrain_transcripts.py`.','']
    if probe:
        lines+=['','## Exploratory self-supervised next-word control','',
          'Added after observing the main study and evaluated on the SAME targets/texts, so this is not independent confirmation. '
          'A separate protocol/source freeze precedes collecting16 new temperature0.8 short completions (cap8 new tokens) per prefix. '
          'Compare their first lexical word to the known first word of the true suffix. This avoids accumulating errors over a whole '
          'continuation. The scorer still sees only text, not the model’s probabilities.',
          '', 'If h of16 samples match, use fixed Jeffreys smoothing p=(h+0.5)/17 and integrate log(p) over prefixes. '
          'This estimates a text-recovery event under the generation policy; it is not the model’s exact token probability. '
          'Candidate text supplies self-supervised word targets, never membership labels. No coefficient/sample-count tuning occurs. '
          'The eight-token cap may censor long words. Different query budgets limit direct efficiency comparisons.',
          '', '| Exploratory output-only score | Membership AUC | TPR at ≤1%FPR | AP |','|---|---:|---:|---:|']
        for m,v in probe_macro.items():lines.append(f'| {m} | {v["auc"]:.4f} | {100*v["tpr_at_1pct_fpr"]:.2f}% | {v["average_precision"]:.4f} |')
        lines+=['',f'An additional exploratory budget diagnostic replays the SAME word-frequency scorer on the original THREE '
          f'full-completion samples, without new queries: membership AUC **{budget_diagnostic["auc"]:.4f}**. '
          'The sixteen-sample short probe changes both sampling budget and horizon; its difference cannot be attributed solely '
          'to shortening. The diagnostic was added after the control results and is not a newly validated primary.']
        lines+=['','The primary next-word control also fits discovery-only GMMs with the same one-vs-two BIC rule, then evaluates '
          'the previously designated held-out texts. This split prevents label fitting but does not undo prior research use of those '
          'texts. No component posterior is called membership probability.','',
          '| Target | Two groups selected? | Held-out membership precision | Recall | Balanced accuracy |','|---|---|---:|---:|---:|']
        for r,p in zip(rs,probe):
            c=p['clustering'];m=c['membership']
            lines.append(f'| {r["corpus"]}/{r["regime"]}/{r["seed"]} | {c["selected_two"]} | {fmt(m["precision"] if m else None)} | {fmt(m["recall"] if m else None)} | {fmt(m["balanced_accuracy"] if m else None)} |')
        lines+=['',f'Additional short-control completions: **{sum(p["query_stats"]["completions"] for p in probe):,}**; '
          f'backend-generated tokens: **{sum(p["query_stats"]["generated_tokens"] for p in probe):,}**.',
          '', 'See [separate control protocol](../../experiments/MEMBRAIN_WORD_PROBE_CONTROL.md). All alternatives are reported; '
          'no best exploratory score is promoted as a newly validated primary.','']
    lines+=['','## Plots','']
    for name in ['membership_comparison','curves_tail_word','curves_full_word','curves_greedy_full',
                 'unsupervised_groups','distance_distributions','greedy_distance_distributions']:
        lines.extend([f'![{name}](plots/{name}.png)',''])
    if probe:lines.extend(['![Exploratory word probe](plots/word_probe_comparison.png)',''])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    example_lines=['# Fixed transcript examples','', 'Rows0,32 and 128; seed8101 repeated condition. Fixed indices, not selected for successful recovery.','']
    for c in ['ag_news','wikitext2']:
        r=next(r for r in group(c,'repeated') if r['seed']==8101)
        with (r['path']/'transcripts.jsonl').open() as transcript_file:
            records=[json.loads(x) for x in transcript_file]
        for idx,label in [(0,'heavily repeated member'),(32,'ordinary member'),(128,'nonmember')]:
            record=records[idx];example_lines += [f'## {c}, row{idx}: {label}','',record['text'],'']
            for q in record['queries']:
                example_lines += [f'Prefix ({q["progress"]:.0%} words): {q["prefix"]}',f'Original suffix: {q["suffix"]}',
                    f'Greedy completion: {q["answer"]["greedy"]}',f'Sample 1: {q["answer"]["samples"][0]}','']
    (root/'EXAMPLES.md').write_text('\n'.join(example_lines)+'\n');print(json.dumps(dict(membership=macro,high_exposure=exposure),indent=2))


if __name__=='__main__':main()
