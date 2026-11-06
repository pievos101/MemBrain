import unittest
import numpy as np
import torch
from spw_transformer import TinyTransformer
from membrain_tail_control import extract,score


class TailTests(unittest.TestCase):
    def test_true_tail_is_invariant_when_only_prefix_partition_changes(self):
        torch.manual_seed(5);m=TinyTransformer(vocab=12,d=16,layers=1,heads=4).eval()
        x=np.random.default_rng(4).integers(3,12,(2,12));c=extract(m,x,[.2,.4,.6],tail=3)
        np.testing.assert_allclose(c['true_embedding'],np.repeat(c['true_embedding'][:,:1],3,1),atol=1e-7)
        np.testing.assert_allclose(c['teacher_ce'],np.repeat(c['teacher_ce'][:,:1],3,1),atol=1e-7)
        s=score(c,[.2,.4,.6]);np.testing.assert_allclose(s['Fixed-tail teacher loss'],-c['teacher_ce'][:,0],atol=1e-6)


if __name__=='__main__':unittest.main()
