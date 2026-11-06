"""Independent implementation of Xie et al., KBS 346 (2026) 116100.

SPW-MIA follows equations (1)--(9) and Algorithm 1. Oracle distributions must
be target-model next-token log probabilities, with context positions reset.
All scores are oriented higher => stronger membership evidence.
"""
from functools import lru_cache
import math
import zlib
import numpy as np


def logsumexp(x):
    x=np.asarray(x,dtype=np.float64);m=x.max()
    return float(m+np.log(np.exp(x-m).sum()))


def softmax(x):
    x=np.asarray(x,dtype=np.float64);return np.exp(x-logsumexp(x))


@lru_cache(maxsize=100000)
def edit_similarity(a,b):
    """Eq. 3, normalized character Levenshtein similarity of decoded tokens."""
    if a==b:return 1.
    if not a or not b:return 0.
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    return (max(len(a),len(b))-previous[-1])/max(len(a),len(b))


def components(target_logp,max_logp,initial_similarity,short_logp,similarity_difference):
    """Paper KL is between normalized POSITION vectors, not vocabulary KL."""
    p=np.asarray(target_logp,dtype=np.float64);q=np.asarray(max_logp,dtype=np.float64)
    lp=p-logsumexp(p);lq=q-logsumexp(q)
    kl=float(np.sum(np.exp(lp)*(lp-lq)))
    weighted_similarity=float(np.dot(initial_similarity,softmax(np.asarray(short_logp)-p)))
    weighted_max=float(np.dot(np.exp(q),softmax(similarity_difference)))
    return dict(kl=kl,weighted_similarity=weighted_similarity,weighted_max=weighted_max,
                spw=weighted_similarity-kl-weighted_max)


def spw_batch(oracle,sequences,teacher_log_probs=None,fixed_length=None,return_token_details=False):
    """Batch equivalent of Algorithm 1, processing independent tokens in parallel.

    Each shortened prefix is the LAST i tokens of the original prefix. Stop at
    the first EXACT equality of edit similarity, even when both are zero.
    Fixed policies force a selected length and accumulate all earlier differences.
    This fixed-policy convention is explicit because the paper does not specify
    separate pseudocode for that ablation.
    """
    x=np.asarray(sequences,dtype=np.int64)
    if x.ndim!=2 or x.shape[1]<3:raise ValueError('Need fixed-length samples with >=3 tokens.')
    if fixed_length is not None and fixed_length!='initial' and int(fixed_length)<1:
        raise ValueError('Fixed prefix length must be positive.')
    n,length=x.shape
    full=oracle.teacher(x) if teacher_log_probs is None else teacher_log_probs
    rows=np.repeat(np.arange(n),length-2);positions=np.tile(np.arange(2,length),n)
    initial=full[:,1:,:].reshape(-1,full.shape[-1])
    target=x[rows,positions];original_lp=initial[np.arange(len(target)),target]
    predicted=initial.argmax(1);maximum_lp=initial[np.arange(len(target)),predicted]
    words=oracle.token_strings
    sim=np.array([edit_similarity(words[int(a)],words[int(b)]) for a,b in zip(target,predicted)])
    active=np.ones(len(target),bool);summed=np.zeros(len(target));selected=np.zeros(len(target));stopped=np.zeros(len(target),int)
    if fixed_length=='initial':desired=positions.copy()
    elif fixed_length is not None:desired=np.minimum(int(fixed_length),positions)
    else:desired=None
    for size in range(1,length):
        ids=np.flatnonzero(active)
        if not len(ids):break
        current_lp=np.empty(len(ids));current_sim=np.empty(len(ids))
        full_ids=positions[ids]<=size
        current_lp[full_ids]=original_lp[ids[full_ids]];current_sim[full_ids]=sim[ids[full_ids]]
        short=np.flatnonzero(~full_ids)
        if len(short):
            subset=ids[short]
            contexts=[tuple(x[r,p-size:p]) for r,p in zip(rows[subset],positions[subset])]
            predictions=oracle.distributions(contexts)
            for j,(idx,dist) in enumerate(zip(subset,predictions)):
                greedy=int(np.argmax(dist));current_lp[short[j]]=dist[target[idx]]
                current_sim[short[j]]=edit_similarity(words[int(target[idx])],words[greedy])
        hit=(current_sim==sim[ids]) if desired is None else (size>=desired[ids])
        hit|=full_ids
        chosen=ids[hit];selected[chosen]=current_lp[hit];stopped[chosen]=size;active[chosen]=False
        missing=ids[~hit];summed[missing]+=np.abs(sim[missing]-current_sim[~hit])
    assert not active.any()
    result=[]
    for i in range(n):
        s=slice(i*(length-2),(i+1)*(length-2))
        c=components(original_lp[s],maximum_lp[s],sim[s],selected[s],summed[s]/positions[s])
        c.update(mean_shortened_ratio=float(np.mean(stopped[s]/positions[s])),
                 mean_initial_similarity=float(sim[s].mean()),stopping_lengths=stopped[s].tolist())
        if return_token_details:
            c.update(target_logp=original_lp[s].tolist(),maximum_logp=maximum_lp[s].tolist(),
                     shortened_logp=selected[s].tolist(),initial_similarity=sim[s].tolist(),
                     similarity_difference=(summed[s]/positions[s]).tolist())
        result.append(c)
    return result


def loss_score(token_logp):return float(np.mean(token_logp))


def min_k_score(token_logp,k=.2):
    p=np.asarray(token_logp);count=max(1,math.ceil(len(p)*k))
    return float(np.sort(p)[:count].mean())


def zlib_score(mean_nll,text):
    size=len(zlib.compress(text.encode('utf-8')))
    return -math.exp(float(mean_nll))/size


def lowercase_score(original_nll,lowercase_nll):
    return math.exp(float(lowercase_nll))-math.exp(float(original_nll))


def neighbor_score(original_nll,neighbor_nlls):return float(np.mean(neighbor_nlls)-original_nll)


def evaluate_scores(labels,scores):
    """Tie-aware ROC/AP, conservative empirical and interpolated TPR@1% FPR."""
    labels=np.asarray(labels,dtype=int);scores=np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all() or set(np.unique(labels))!={0,1}:raise ValueError('Invalid binary evaluation input.')
    order=np.argsort(-scores,kind='stable');y=labels[order];s=scores[order]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.r_[0,np.cumsum(y)[ends]];fp=np.r_[0,np.cumsum(1-y)[ends]]
    recall=tp/labels.sum();fpr=fp/(len(labels)-labels.sum())
    precision=np.divide(tp,tp+fp,out=np.ones(len(tp),float),where=(tp+fp)>0)
    # Collapse vertical ROC segments only for the interpolated statistic.
    u=np.unique(fpr);upper=np.array([recall[fpr==x].max() for x in u])
    return dict(auc=float(np.trapezoid(recall,fpr)),
                tpr_at_1pct_fpr=float(recall[fpr<=.01+1e-12].max()),
                tpr_at_1pct_fpr_interpolated=float(np.interp(.01,u,upper)),
                average_precision=float(np.sum(np.diff(recall)*precision[1:])),
                auprc_trapezoid=float(np.trapezoid(precision,recall)))


def component_ablations(c):
    a,k,b=c['weighted_similarity'],c['kl'],c['weighted_max']
    return {'-KL':-k,'Similarity':a,'-Max':-b,'Similarity-KL':a-k,
            '-KL-Max':-k-b,'Similarity-Max':a-b,'SPW-MIA':a-k-b}
