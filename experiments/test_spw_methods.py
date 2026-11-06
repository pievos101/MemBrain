import unittest
import numpy as np
import torch
from spw_transformer import TinyTransformer,Oracle
from spw_methods import (spw_batch,components,edit_similarity,evaluate_scores,
                         min_k_score,lowercase_score,neighbor_score)


class ImplementationTests(unittest.TestCase):
    def test_cached_generation_matches_full_context(self):
        torch.manual_seed(7);model=TinyTransformer(vocab=17,d=16,layers=2,heads=4).eval()
        x=torch.randint(3,17,(3,9))
        with torch.no_grad():
            full,h,_=model(x,return_hidden=True)
            _,_,cache=model(x[:,:4],return_hidden=True)
            for i in range(4,9):
                step,hidden,cache=model(x[:,i:i+1],cache=cache,return_hidden=True)
                torch.testing.assert_close(step[:,0],full[:,i],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(hidden[:,0],h[:,i],atol=2e-6,rtol=2e-5)

    def test_causal_outputs_do_not_read_future_tokens(self):
        torch.manual_seed(9);model=TinyTransformer(vocab=17,d=16,layers=2,heads=4).eval()
        x=torch.randint(3,17,(2,8));changed=x.clone();changed[:,4:]=3
        with torch.no_grad():
            torch.testing.assert_close(model(x)[:,:4],model(changed)[:,:4])

    def test_spw_against_unbatched_reference(self):
        torch.manual_seed(2);model=TinyTransformer(vocab=11,d=16,layers=1,heads=4).eval()
        words=['a','bc','de','apple','ape','cat','cart','dog','dot','bird','fish']
        oracle=Oracle(model,words);x=np.array([[3,4,6,8,5,9],[7,4,3,5,8,10]])
        got=spw_batch(oracle,x)
        # Literal sequential transcription, not using batched stopping logic.
        for sample,result in zip(x,got):
            pt=[];pm=[];sims=[];ps=[];diffs=[];stops=[]
            for t in range(len(sample)-1,1,-1):
                full=oracle.distributions([tuple(sample[:t])])[0];y=sample[t]
                initial=edit_similarity(words[full.argmax()],words[y]);diff=0
                for size in range(1,t+1):
                    pred=oracle.distributions([tuple(sample[t-size:t])])[0]
                    similarity=edit_similarity(words[pred.argmax()],words[y])
                    if similarity==initial:break
                    diff+=abs(initial-similarity)
                pt.append(full[y]);pm.append(full.max());sims.append(initial)
                ps.append(pred[y]);diffs.append(diff/t);stops.append(size)
            expected=components(pt,pm,sims,ps,diffs)
            for k,v in expected.items():self.assertAlmostEqual(v,result[k],places=6)
            self.assertEqual(stops[::-1],result['stopping_lengths'])

    def test_zero_similarity_stops_and_uses_suffix_context(self):
        class Mock:
            token_strings=['a','b','c','d'];contexts=[]
            def teacher(self,x):return np.log(np.tile([.7,.1,.1,.1],(1,3,1)))
            def distributions(self,cs):
                self.contexts.extend(cs);return [np.log([.7,.1,.1,.1]) for c in cs]
        oracle=Mock();r=spw_batch(oracle,[[1,2,3,1]])[0]
        self.assertEqual(r['stopping_lengths'],[1,1])
        self.assertEqual(oracle.contexts,[(2,),(3,)])

    def test_ties_and_low_fpr_are_conservative(self):
        r=evaluate_scores([0,0,1,1],[1,1,1,1])
        self.assertEqual(r['auc'],.5);self.assertEqual(r['average_precision'],.5)
        self.assertEqual(r['tpr_at_1pct_fpr'],0)
        perfect=evaluate_scores([0,0,1,1],[0,0,1,1])
        self.assertEqual(perfect['auc'],1);self.assertEqual(perfect['average_precision'],1)
        mixed=evaluate_scores([0,1,0,1],[.1,.4,.4,.8])
        self.assertAlmostEqual(mixed['auc'],.875)
        self.assertAlmostEqual(mixed['average_precision'],5/6)

    def test_nonmatching_similarities_accumulate_before_stop_only(self):
        class Mock:
            token_strings=['aa','ab','bb']
            def teacher(self,x):
                return np.log([[[.7,.2,.1],[.1,.2,.7],[.7,.2,.1]]])
            def distributions(self,contexts):
                table={(1,):[.7,.2,.1],(2,):[.1,.2,.7],(1,2):[.2,.7,.1]}
                return [np.log(table[c]) for c in contexts]
        result=spw_batch(Mock(),[[0,1,2,0]])[0]
        expected=components(np.log([.7,.7]),np.log([.7,.7]),[1,1],np.log([.7,.7]),[.5,.5])
        self.assertEqual(result['stopping_lengths'],[2,3])
        for k,v in expected.items():self.assertAlmostEqual(result[k],v)

    def test_formulas(self):
        self.assertAlmostEqual(edit_similarity('kitten','sitting'),4/7)
        self.assertEqual(min_k_score([-1,-2,-3,-4,-5]),-5)
        self.assertEqual(neighbor_score(2,[3,5]),2)
        self.assertAlmostEqual(lowercase_score(np.log(2),np.log(5)),3)
        c=components(np.log([.2,.2]),np.log([.4,.4]),[1,0],np.log([.4,.2]),[0,0])
        self.assertAlmostEqual(c['kl'],0);self.assertAlmostEqual(c['weighted_similarity'],2/3)
        self.assertAlmostEqual(c['weighted_max'],.4)


if __name__=='__main__':unittest.main()
