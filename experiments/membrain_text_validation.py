"""Confirmation helpers. Only strings and unlabeled scores enter this module."""
import copy
import numpy as np
from membrain_text import query_plan,similarity,area,words
from membrain_word_probe import score_record


def collect(texts,complete,samples,max_new_tokens,seed):
    records=[dict(text=t,queries=[{k:q[k] for k in ['prefix','suffix','progress']} for q in query_plan(t)]) for t in texts]
    prompts=[q['prefix'] for r in records for q in r['queries']]
    answers=complete(prompts,samples=samples,temperature=.8,max_new_tokens=max_new_tokens,seed=seed)
    if len(prompts)!=len(answers):raise ValueError('Completion count mismatch')
    i=0
    for r in records:
        for q in r['queries']:q['answer']=dict(greedy=answers[i]['greedy'],samples=list(answers[i]['samples']));i+=1
    return records


def combined_scores(short,replicate,long):
    results=[]
    for a,b,c in zip(short,replicate,long):
        if a['text']!=b['text'] or a['text']!=c['text']:raise ValueError('Candidate mismatch')
        pooled=copy.deepcopy(a);budget3=copy.deepcopy(a)
        for qa,qb,qp,q3 in zip(a['queries'],b['queries'],pooled['queries'],budget3['queries']):
            if qa['prefix']!=qb['prefix']:raise ValueError('Query mismatch')
            qp['answer']['samples']+=qb['answer']['samples'];q3['answer']['samples']=q3['answer']['samples'][:3]
        sa,_=score_record(a);sb,_=score_record(b);sp,_=score_record(pooled);s3,_=score_record(budget3)
        progress=[q['progress'] for q in c['queries']]
        results.append({'Membrain 16':sa['Membrain next-word probe'],'Membrain replicate16':sb['Membrain next-word probe'],
            'Membrain pooled32':sp['Membrain next-word probe'],'Membrain nested3':s3['Membrain next-word probe'],
            'Hard-prefix word probe':sa['Hard-prefix word probe'],'Greedy next-word':sa['Greedy next-word recovery'],
            'Greedy whole-suffix':area([similarity(q['suffix'],q['answer']['greedy'])['word_edit'] for q in c['queries']],progress),
            'Greedy fixed-tail':area([similarity(q['suffix'],q['answer']['greedy'],tail=4)['word_edit'] for q in c['queries']],progress),
            'Length-only control':-len(words(a['text']))})
    return {key:np.array([r[key] for r in results]) for key in results[0]}


def rank_correlation(a,b):
    def ranks(x):
        order=np.argsort(x,kind='stable');v=np.asarray(x)[order];starts=np.r_[0,np.flatnonzero(np.diff(v))+1];ends=np.r_[starts[1:],len(v)]
        out=np.zeros(len(v));out[order]=np.repeat((starts+ends-1)/2,ends-starts);return out
    x,y=ranks(a),ranks(b)
    return float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None
