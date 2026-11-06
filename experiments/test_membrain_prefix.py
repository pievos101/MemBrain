import unittest
import numpy as np
import torch
from torch.nn import functional as F
from membrain_prefix import (token_weights,recovery_features,assemble_features,feature_mask,
                             fit_logistic,decision,calibrate,probabilities,fpr_threshold)


class ConstantLM(torch.nn.Module):
    def forward(self,ids,cache=None,return_hidden=False):
        logits=torch.zeros(*ids.shape,8);logits[:,:,3]=5
        h=F.one_hot(ids,8).float()
        if return_hidden:return logits,h,[]
        return logits


class MembrainPrefixTests(unittest.TestCase):
    def test_weights_are_suffix_normalized_and_uniform_without_gain(self):
        w=token_weights([[1,1,1],[0,0,4]],np.zeros((2,3)))
        np.testing.assert_allclose(w.sum(1),1)
        np.testing.assert_allclose(w[0],[1/3]*3)
        self.assertGreater(w[1,2],w[1,0]);self.assertTrue((w>=1/6).all())

    def test_alignment_and_free_running_ce_target_true_tokens(self):
        x=np.full((2,8),3);x[1,-1]=4
        detail={'target_logp':[0.]*6,'shortened_logp':[0.]*5+[-4.]}
        curves,a,b=recovery_features(ConstantLM(),x,[detail,detail],[.25,.5,.75])
        np.testing.assert_allclose(curves['aligned_cosine'][0],1)
        np.testing.assert_allclose(curves['aligned_cosine'][1],[5/6,3/4,1/2],atol=1e-6)
        self.assertTrue((curves['weighted_cosine'][1]<curves['aligned_cosine'][1]).all())
        np.testing.assert_array_equal(curves['exact'][0],1)
        np.testing.assert_array_equal(curves['exact'][1],0)
        np.testing.assert_allclose(curves['rollout_excess_ce'],0,atol=1e-6)
        logits=torch.zeros(8);logits[3]=5
        good=-logits.log_softmax(0)[3].item();bad=-logits.log_softmax(0)[4].item()
        np.testing.assert_allclose(curves['rollout_ce'][1],[(5*good+bad)/6,(3*good+bad)/4,(good+bad)/2],atol=1e-6)
        self.assertEqual(a.shape,(2,3,8));self.assertEqual(b.shape,(2,3,8))

    def test_normalization_is_fixed_from_shadow_training(self):
        x=np.array([[-2.,1],[-1.,2],[1.,3],[2.,4]])
        y=np.array([0,0,1,1]);m=fit_logistic(x,y,.1)
        np.testing.assert_array_equal(m['mu'],x.mean(0))
        before=decision(m,x[:1])[0]
        after=decision(m,np.r_[x[:1],[[10000.,-10000.]]])[0]
        self.assertEqual(before,after)
        self.assertGreater(decision(m,x)[-1],decision(m,x)[0])

    def test_old_pooled_trajectories_are_retained(self):
        from spw_transformer import TinyTransformer,trajectories
        torch.manual_seed(83);model=TinyTransformer(vocab=12,d=16,layers=1,heads=4).eval()
        x=np.random.default_rng(1).integers(3,12,(3,8));fractions=[.25,.5,.75]
        details=[dict(target_logp=[0.]*6,shortened_logp=[0.]*6) for _ in x]
        old=trajectories(model,x,fractions);curves,a,b=recovery_features(model,x,details,fractions)
        np.testing.assert_allclose(a,old['true'],atol=1e-6)
        np.testing.assert_allclose(b,old['generated'],atol=1e-6)
        for key in ['accuracy','exact','rollout_ce']:np.testing.assert_allclose(curves[key],old[key],atol=1e-6)

    def test_ablations_remove_feature_groups(self):
        groups=['embedding','probability','generation']
        np.testing.assert_array_equal(feature_mask(groups,'Probability-only learner'),[False,True,False])
        np.testing.assert_array_equal(feature_mask(groups,'Membrain curves only'),[True,False,True])
        np.testing.assert_array_equal(feature_mask(groups,'Membrain without embeddings'),[False,True,True])

    def test_threshold_does_not_allow_extra_false_positives_on_ties(self):
        nonmember=np.array([1.,1.,.5,.1])
        t=fpr_threshold(nonmember,fpr=.25)
        self.assertLessEqual(np.mean(nonmember>=t),.25)
        t=fpr_threshold(nonmember,fpr=0)
        self.assertEqual(np.mean(nonmember>=t),0)

    def test_positive_slope_calibration_preserves_ranking(self):
        logits=np.linspace(-3,3,30);y=(logits>0).astype(int)
        cal=calibrate(logits,y);p=probabilities(cal,logits)
        self.assertGreater(cal['slope'],0);self.assertTrue((np.diff(p)>=0).all())
        self.assertTrue(((p>0)&(p<=1)).all())


if __name__=='__main__':unittest.main()
