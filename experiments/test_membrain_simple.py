import unittest
import inspect
import numpy as np
import torch
from membrain_simple import extract,score
from spw_transformer import TinyTransformer,trajectories


class SimpleTests(unittest.TestCase):
    def test_retains_original_two_embedding_vectors(self):
        torch.manual_seed(3);model=TinyTransformer(vocab=12,d=16,layers=1,heads=4).eval()
        x=np.random.default_rng(2).integers(3,12,(3,8));f=[.25,.5,.75]
        got=extract(model,x,f);old=trajectories(model,x,f)
        np.testing.assert_allclose(got['true_embedding'],old['true'],atol=1e-6)
        np.testing.assert_allclose(got['generated_embedding'],old['generated'],atol=1e-6)
        np.testing.assert_allclose(got['rollout_ce'],old['rollout_ce'],atol=1e-6)
        np.testing.assert_allclose(got['accuracy'],old['accuracy'],atol=1e-6)

    def test_fixed_cost_and_no_membership_input(self):
        self.assertEqual(list(inspect.signature(score).parameters),['curves','fractions'])
        c=dict(cosine=np.array([[1.,1.,1.],[-1.,-1.,-1.]]),teacher_ce=np.full((2,3),2.),rollout_ce=np.full((2,3),3.))
        s=score(c,[.25,.5,.75]);np.testing.assert_allclose(s['Membrain'],[-2,-5])
        np.testing.assert_allclose(s['Teacher loss curve'],[-2,-2])
        one=score({k:v[:1] for k,v in c.items()},[.25,.5,.75])
        self.assertEqual(one['Membrain'][0],s['Membrain'][0])


if __name__=='__main__':unittest.main()
