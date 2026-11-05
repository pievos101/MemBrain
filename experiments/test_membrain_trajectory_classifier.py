import unittest
import numpy as np
import torch
import membrain_trajectory_classifier as m


class TrajectoryClassifierChecks(unittest.TestCase):
    def test_auxiliary_sentences_are_disjoint(self):
        aux=m.auxiliary_sentences();evaluation,background=m.recovery.sentence_pool()
        self.assertEqual(len(set(aux)),320)
        self.assertFalse(set(aux)&set(evaluation+background))

    def test_exact_generated_suffix_has_same_trajectory(self):
        torch.set_num_threads(2);torch.manual_seed(1)
        lm=m.recovery.SentenceLM()
        for p in lm.parameters():p.data.zero_()
        lm.output.bias.data[ord('A')-31]=50
        r,_=m.recovery.recover(lm,['A'*80],2,return_embeddings=True)
        self.assertEqual(r['original_trajectory'].shape,(1,12,96))
        self.assertEqual(r['generated_trajectories'].shape,(1,3,12,96))
        self.assertTrue(np.allclose(r['original_trajectory'][:,None],r['generated_trajectories'],atol=1e-6))

    def test_class_direction_and_saved_prediction(self):
        torch.set_num_threads(2);rng=np.random.default_rng(2)
        o=(rng.normal(size=(320,12,96))*.1-2).astype('float32')
        g=(rng.normal(size=(320,9,12,96))*.1+2).astype('float32')
        checkpoint=m.fit_classifier(o,g,'single90',51)
        po,pg=m.predict(checkpoint,o[240:],g[240:,1:])
        self.assertLess(float(po.mean()),.1)
        self.assertGreater(float(pg.mean()),.9)
        same=np.repeat(o[240:,None],2,axis=1)
        a,b=m.predict(checkpoint,o[240:],same)
        self.assertTrue(np.allclose(a[:,None],b,atol=1e-6))


if __name__=='__main__':unittest.main()
