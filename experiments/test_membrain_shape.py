import unittest
import numpy as np
import torch
from membrain_shape import cos,geometry,extract,score
from membrain_simple import extract as prior_extract
from spw_transformer import TinyTransformer


class ShapeTests(unittest.TestCase):
    def test_identical_shapes_have_perfect_agreement(self):
        a=np.random.default_rng(3).normal(size=(2,10,8));g=geometry(a,a)
        for k in ['shape','centered','velocity','proximity']:np.testing.assert_allclose(g[k],1,atol=1e-10)

    def test_zero_directions_provide_no_evidence(self):
        np.testing.assert_array_equal(cos(np.zeros((2,4)),np.ones((2,4))),0)
        a=np.ones((2,10,4));g=geometry(a,a)
        np.testing.assert_allclose(g['centered'],0);np.testing.assert_allclose(g['velocity'],0)

    def test_scores_do_not_depend_on_other_candidates(self):
        rng=np.random.default_rng(7);c={k:rng.normal(size=(3,10,8)) for k in ['true_embedding','generated_embedding','hard_true_embedding','hard_generated_embedding']}
        for k in ['teacher_ce','rollout_ce','hard_teacher_ce','hard_rollout_ce']:c[k]=rng.uniform(1,8,(3,10))
        lp=rng.uniform(-8,-1,(3,47));all_scores=score(c,lp);one=score({k:v[:1] for k,v in c.items()},lp[:1])
        for k in all_scores:np.testing.assert_allclose(one[k],all_scores[k][:1])
        self.assertTrue((np.abs(all_scores['Membrain shape hybrid']-all_scores['Min-k20%'])<=.5+1e-12).all())

    def test_uniform_embeddings_and_loss_preserved(self):
        torch.manual_seed(5);model=TinyTransformer(vocab=12,d=16,layers=1,heads=4).eval()
        x=np.random.default_rng(3).integers(3,12,(2,8));f=[.25,.5,.75]
        got=extract(model,x,f);prior=prior_extract(model,x,f)
        for k in ['true_embedding','generated_embedding','teacher_ce','rollout_ce']:np.testing.assert_allclose(got[k],prior[k],atol=1e-6)
        self.assertTrue((got['hard_teacher_ce']>=got['teacher_ce']-1e-6).all())


if __name__=='__main__':unittest.main()
