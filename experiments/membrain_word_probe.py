"""Text-only self-supervised next-word recovery; no model probability inputs."""
import numpy as np
from membrain_text import words,area


def score_record(record):
    progress=[];logp=[];greedy=[];frequencies=[]
    for q in record['queries']:
        target=words(q['suffix'])[0];samples=q['answer']['samples']
        hits=sum(bool(words(s)) and words(s)[0]==target for s in samples)
        probability=(hits+.5)/(len(samples)+1) # Fixed Jeffreys smoothing, no labels.
        progress.append(q['progress']);logp.append(float(np.log(probability)));frequencies.append(hits/len(samples))
        g=words(q['answer']['greedy']);greedy.append(float(bool(g) and g[0]==target))
    k=max(1,int(np.ceil(.2*len(logp))))
    return {'Membrain next-word probe':area(logp,progress),'Hard-prefix word probe':float(np.mean(np.sort(logp)[:k])),
      'Greedy next-word recovery':area(greedy,progress),'Sampled next-word recovery':area(frequencies,progress)},dict(progress=progress,log_recovery=logp,frequencies=frequencies)
