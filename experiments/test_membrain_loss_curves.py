import unittest
import numpy as np
import torch
from torch.nn import functional as F
import membrain_loss_curves as m


class LossCurveChecks(unittest.TestCase):
    def test_fast_auc_matches_pairwise_with_ties(self):
        rng=np.random.default_rng(3)
        for _ in range(10):
            a,b=rng.integers(0,8,50),rng.integers(0,8,100)
            self.assertAlmostEqual(m.auc(a,b),m.base.auc(a,b))
    def test_uniform_model_and_alignment(self):
        torch.set_num_threads(2);torch.manual_seed(1)
        lm=m.base.LM();docs=torch.randint(53,(3,48))
        o,g,t=m.loss_curves(lm,docs,2)
        with torch.no_grad():
            expected=F.cross_entropy(lm(docs[:,:-1]).transpose(1,2),docs[:,1:],reduction='none').numpy()
        self.assertTrue(np.allclose(t,expected))
        for j,p in enumerate(m.PREFIXES):
            self.assertTrue(np.allclose(o[:,j],expected[:,p-1:p+7].mean(1)))
        for p in lm.parameters():p.data.zero_()
        o,g,t=m.loss_curves(lm,docs,2)
        self.assertEqual(g.shape,(3,2,4))
        self.assertTrue(np.allclose(o,np.log(53)))
        self.assertTrue(np.allclose(g,np.log(53)))

    def test_corpus_disjoint_and_difficulty_balanced(self):
        x,d=m.corpus(11,True)
        self.assertEqual(len({tuple(r.tolist()) for r in x}),1120)
        for a,b in [(0,400),(400,440),(440,840),(840,1040),(1040,1120)]:
            self.assertEqual(d[a:b].sum(),(b-a)//2)
        self.assertEqual(len(m.EVAL_IDS),640)
        self.assertEqual(len(set(m.EVAL_IDS)),640)
        self.assertEqual(m.LABELS.sum(),240)
        self.assertFalse(set(m.TRAIN_SEEDS)&set(m.CAL_SEEDS+m.TEST_SEEDS))
        self.assertFalse(set(m.CAL_SEEDS)&set(m.TEST_SEEDS))

    def test_natural_text_source_and_unique_blocks(self):
        x,d=m.corpus(43001,False,True)
        self.assertEqual(x.shape,(1120,48))
        self.assertEqual(len({tuple(r.tolist()) for r in x}),1120)
        self.assertTrue(torch.all((x>=0)&(x<53)))
        self.assertTrue(np.all(d==-1))

    def test_information_matched_baselines(self):
        o=np.arange(2560).reshape(640,4)/100
        r=dict(original=o,generated=np.repeat((o+1)[:,None,:],8,axis=1),
               auxiliary_gap=np.ones(640),token_losses=np.ones((640,47))*2)
        self.assertTrue(np.allclose(m.design(r,'Generated-reference gap'),1))
        self.assertTrue(np.allclose(m.design(r,'LOSS same windows')[:,0],-o.mean(1)))
        self.assertEqual(m.design(r,'Membrain paired curves').shape,(640,8))
        self.assertEqual(m.design(r,'Original curve MLP').shape,(640,4))


if __name__=='__main__':unittest.main()
