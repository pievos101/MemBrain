"""Output-only Membrain. Inputs are strings/completions; no model internals.

Optional Gaussian mixtures fit UNLABELED recovery scores. Component posterior
means high-recovery component probability, never calibrated membership probability.
"""
import re
import unicodedata
import numpy as np
from rapidfuzz.distance import Levenshtein,LCSseq

STAGES=(.05,.1,.2,.35,.5,.65,.8,1.)
WORD=re.compile(r"\w+(?:['’\-]\w+)*",re.UNICODE)
SAMPLES=3


def words(text):return [m.group().casefold() for m in WORD.finditer(unicodedata.normalize('NFC',text))]


def similarity(reference,completion,tail=None):
    a,b=words(reference),words(completion)
    if tail is not None:a,b=a[-tail:],b[-tail:]
    if not a:return dict(word_edit=1. if not b else 0.,char_edit=1. if not b else 0.,rouge_l=1. if not b else 0.,exact=float(not b))
    return dict(word_edit=1-float(Levenshtein.normalized_distance(a,b)),
        char_edit=1-float(Levenshtein.normalized_distance(' '.join(a),' '.join(b))),
        rouge_l=2*LCSseq.similarity(a,b)/max(1,len(a)+len(b)),exact=float(a==b))


def query_plan(text):
    spans=list(WORD.finditer(text));n=len(spans)
    if n<12:raise ValueError('Need >=12 lexical words for this fixed-tail benchmark')
    positions=sorted(set(max(1,min(n-4,round(r*(n-4)))) for r in STAGES))
    result=[]
    for p in positions:
        boundary=spans[p-1].end();prefix=text[:boundary]
        result.append(dict(prefix=prefix,suffix=text[boundary:].strip(),progress=p/n,
            null_prefix=' '.join(reversed(prefix.split()))))
    return result


def collect(texts,complete,seed=1):
    """complete sees ONLY prefix strings and generation settings, never suffixes."""
    records=[dict(text=t,queries=query_plan(t)) for t in texts];prefixes=[]
    for r in records:
        for q in r['queries']:prefixes.extend([q['prefix'],q['null_prefix']])
    answers=complete(prefixes,samples=SAMPLES,temperature=.8,max_new_tokens=96,seed=seed)
    if len(answers)!=len(prefixes):raise ValueError('Completion count mismatch')
    at=0
    for r in records:
        for q in r['queries']:
            for key in ['answer','null_answer']:
                a=answers[at];at+=1
                if not isinstance(a['greedy'],str) or len(a['samples'])!=SAMPLES or not all(isinstance(x,str) for x in a['samples']):raise ValueError('Only text answers allowed')
                q[key]=a
    return records


def area(values,progress):
    r=np.asarray(progress);v=np.asarray(values);density=1-r
    return float(np.trapezoid(v*density,r)/np.trapezoid(density,r))


def score_record(record):
    curves={k:[] for k in ['tail_word','tail_char','tail_rouge','tail_exact','tail_q25','full_word','greedy_full','greedy_tail','null_tail']}
    distributions=[];progress=[]
    for q in record['queries']:
        progress.append(q['progress']);tails=[similarity(q['suffix'],c,tail=4) for c in q['answer']['samples']]
        full=[similarity(q['suffix'],c) for c in q['answer']['samples']]
        null=[similarity(q['suffix'],c,tail=4)['word_edit'] for c in q['null_answer']['samples']]
        dist=[1-t['word_edit'] for t in tails];distributions.append(dist)
        for key,metric in [('tail_word','word_edit'),('tail_char','char_edit'),('tail_rouge','rouge_l'),('tail_exact','exact')]:curves[key].append(float(np.mean([t[metric] for t in tails])))
        curves['tail_q25'].append(float(np.quantile([t['word_edit'] for t in tails],.25)))
        curves['full_word'].append(float(np.mean([t['word_edit'] for t in full])))
        curves['greedy_full'].append(similarity(q['suffix'],q['answer']['greedy'])['word_edit'])
        curves['greedy_tail'].append(similarity(q['suffix'],q['answer']['greedy'],tail=4)['word_edit'])
        curves['null_tail'].append(float(np.mean(null)))
    a={k:area(v,progress) for k,v in curves.items()}
    speed=0.
    for i in range(len(progress)):
        if min(curves['tail_word'][i:])>=.75:speed=1-progress[i];break
    scores={'Membrain text':a['tail_word'],'Membrain prefix advantage':a['tail_word']-a['null_tail'],
       'Sampled whole-suffix edit':a['full_word'],'Greedy whole-suffix edit':a['greedy_full'],
       'Greedy fixed-tail edit':a['greedy_tail'],'Sampled tail character edit':a['tail_char'],
       'Sampled tail ROUGE-L':a['tail_rouge'],'Sampled tail exact recovery':a['tail_exact'],
       'Conservative recovery quantile':a['tail_q25'],'Sustained recovery speed':speed,
       'Single longest-prefix recovery':curves['tail_word'][-1]}
    return scores,dict(progress=progress,curves=curves,word_distance_samples=distributions)


def fit_mixture(scores):
    """1 vs2 components by BIC; deterministic starts, no class labels/known priors."""
    x=np.asarray(scores,dtype=float);floor=1e-4
    if len(x)<20 or not np.isfinite(x).all():raise ValueError('Need >=20 finite unlabeled scores')
    mean=x.mean();var=max(x.var(),floor)
    ll1=float(np.sum(-.5*(np.log(2*np.pi*var)+(x-mean)**2/var)))
    best=None
    for quantiles in [[.1,.9],[.25,.75],[.05,.95]]:
        mu=np.quantile(x,quantiles);variance=np.full(2,var);weight=np.full(2,.5);previous=-np.inf
        for _ in range(200):
            logp=np.log(weight)[None,:]-.5*(np.log(2*np.pi*variance)[None,:]+(x[:,None]-mu[None,:])**2/variance[None,:])
            m=logp.max(1);normal=m+np.log(np.exp(logp-m[:,None]).sum(1));ll=float(normal.sum())
            resp=np.exp(logp-normal[:,None]);mass=resp.sum(0).clip(1e-8)
            mu=(resp*x[:,None]).sum(0)/mass
            variance=np.maximum((resp*(x[:,None]-mu[None,:])**2).sum(0)/mass,floor)
            weight=np.clip(mass/len(x),.05,.95);weight/=weight.sum()
            if abs(ll-previous)<1e-7:break
            previous=ll
        # Evaluate the final parameters, not the preceding EM iterate.
        logp=np.log(weight)[None,:]-.5*(np.log(2*np.pi*variance)[None,:]+(x[:,None]-mu)**2/variance)
        m=logp.max(1);ll=float(np.sum(m+np.log(np.exp(logp-m[:,None]).sum(1))))
        if best is None or ll>best[0]:best=(ll,mu.copy(),variance.copy(),weight.copy())
    ll,mu,variance,weight=best;order=np.argsort(mu);bic1=2*np.log(len(x))-2*ll1;bic2=5*np.log(len(x))-2*ll
    return dict(means=mu[order].tolist(),variances=variance[order].tolist(),weights=weight[order].tolist(),
        bic_one=float(bic1),bic_two=float(bic2),selected_two=bool(bic2<bic1),fit_count=len(x),variance_floor=floor)


def component_probability(model,scores):
    x=np.asarray(scores,dtype=float);mu=np.array(model['means']);v=np.array(model['variances']);w=np.array(model['weights'])
    logp=np.log(w)[None,:]-.5*(np.log(2*np.pi*v)[None,:]+(x[:,None]-mu)**2/v)
    p=np.exp(logp-logp.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    return p[:,1] # High recovery by construction; never label-based orientation.
