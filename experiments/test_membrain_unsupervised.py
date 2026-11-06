import inspect
import unittest
import numpy as np
import torch
from torch.nn import functional as F
import membrain_unsupervised as method


class ConstantLM(torch.nn.Module):
    def forward(self,ids,cache=None,return_hidden=False):
        logits=torch.zeros(*ids.shape,8);logits[:,:,3]=5;h=F.one_hot(ids,8).float()
        return (logits,h,[]) if return_hidden else logits


class UnsupervisedTests(unittest.TestCase):
    def test_score_has_no_label_or_fitting_input(self):
        self.assertEqual(list(inspect.signature(method.score_curves).parameters),['curves','token_logp','fractions'])
        self.assertFalse(hasattr(method,'fit_logistic'))

    def test_recovery_support_is_monotone_in_alignment_and_degradation(self):
        r,_,_=method.recovery_support(np.array([0.,.5,1.]),np.zeros(3),np.ones(3)*2,np.ones(3)*2)
        np.testing.assert_allclose(r,[.5,.75,1])
        r,_,_=method.recovery_support(np.ones(3),np.zeros(3),np.array([2.,5.,8.]),np.ones(3)*2)
        np.testing.assert_allclose(r,[1,.5,1/3])

    def test_null_removes_collapsed_embedding_similarity(self):
        np.testing.assert_allclose(method.corrected_alignment([1,.8,.2],[1,.6,.4]),[0,.5,0])

    def test_earlier_recovery_receives_more_weight(self):
        early=np.zeros((1,10));early[:,:4]=1;late=early[:,::-1]
        self.assertGreater(method.early_area(early)[0],method.early_area(late)[0])
        np.testing.assert_allclose(method.early_area(np.ones((2,10))),1)

    def test_batch_independent_scores_and_min_k_anchor(self):
        rng=np.random.default_rng(5);n=3
        curves={k:rng.uniform(0,.8,(n,10)) for k in ['aligned','null','uniform_aligned','uniform_null']}
        for k in ['teacher_ce','rollout_ce','uniform_teacher_ce','uniform_rollout_ce']:curves[k]=rng.uniform(1,6,(n,10))
        lp=rng.uniform(-8,-1,(n,47));scores=method.score_curves(curves,lp)
        single=method.score_curves({k:v[:1] for k,v in curves.items()},lp[:1])
        for k in scores:np.testing.assert_allclose(single[k],scores[k][:1])
        self.assertTrue((scores['Membrain']<=scores['Min-k20%']+1e-10).all())
        np.testing.assert_allclose(scores['Min-k20%'],np.sort(lp,axis=1)[:,:10].mean(1))

    def test_rollout_ce_scores_true_not_generated_tokens(self):
        x=np.full((2,8),3);x[1,-1]=4;details=[dict(target_logp=[0.]*6,shortened_logp=[0.]*6)]*2
        c=method.extract_curves(ConstantLM(),x,details,fractions=[.25,.5,.75])
        logits=torch.zeros(8);logits[3]=5;lp=logits.log_softmax(0)
        good=-lp[3].item();bad=-lp[4].item()
        np.testing.assert_allclose(c['rollout_ce'][1],[(5*good+bad)/6,(3*good+bad)/4,(good+bad)/2],atol=1e-6)
        np.testing.assert_allclose(c['accuracy'][1],[5/6,3/4,1/2],atol=1e-6)
        np.testing.assert_allclose(c['teacher_ce'],c['rollout_ce'],atol=1e-6)


if __name__=='__main__':unittest.main()
