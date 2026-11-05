import unittest
import numpy as np
import torch
import membrain_geometry as m


class MembrainChecks(unittest.TestCase):
    def test_region_class_direction_on_separable_data(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        o=torch.randn(340,4,48)*.1-2
        g=torch.randn(340,2,4,48)*.1+2
        lo,lg,accuracy=m.region_classifier(o,g)
        self.assertTrue(np.all(lo<0))
        self.assertTrue(np.all(lg>0))
        self.assertTrue(min(accuracy)>.95)

    def test_direction_learned_from_shadow_labels(self):
        # Here membership deliberately decreases the region feature.
        x=np.array([[-3.],[-2.],[2.],[3.]])
        y=np.array([1.,1.,0.,0.])
        w,b=m.logistic(x,y)
        self.assertLess(w[0],0)
        self.assertTrue((m.sigmoid(x@w+b)[:2]>.5).all())

    def test_calibration_preserves_ranking(self):
        x=np.array([-2.,-1.,0.,1.,2.])
        a,b=m.calibrate(x,np.array([0.,0.,1.,1.,1.]))
        self.assertGreater(a,0)
        self.assertTrue((np.diff(m.sigmoid(a*x+b))>0).all())

    def test_paired_features(self):
        r=dict(original_logits=np.ones((200,4))*3,
               generated_logits=np.ones((200,16,4)),loss=np.ones(200))
        self.assertTrue(np.all(m.matrix(r,'Membrain mean gap')==2))
        self.assertEqual(m.matrix(r,'Membrain prefix gaps').shape,(200,4))
        self.assertEqual(m.ece(np.array([.25,.25,.75,.75]),np.array([0,0,1,1])),.25)


if __name__=='__main__': unittest.main()
