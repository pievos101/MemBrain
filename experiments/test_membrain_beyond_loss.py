import unittest
import numpy as np
import torch
import membrain_beyond_loss as m


class BeyondLossChecks(unittest.TestCase):
    def test_anchor_coordinates_are_rotation_invariant(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(5,32));a=rng.normal(size=(32,32))
        rotation=np.linalg.qr(rng.normal(size=(32,32)))[0]
        self.assertTrue(np.allclose(m.anchor_geometry(x,a),m.anchor_geometry(x@rotation,a@rotation)))
    def test_fresh_disjoint_role_pools(self):
        pools,excluded=m.fresh_pools()
        self.assertEqual(excluded,576)
        self.assertEqual(len(set(sum(pools.values(),[]))),1216)
        self.assertEqual([len(pools[k]) for k in ['train','validation','target']],[256]*3)

    def test_parameter_and_order_controls(self):
        count=lambda model:sum(p.numel() for p in model.parameters())
        cnn,flat,shuffle=[m.Network(v) for v in ['cnn','flat','shuffle']]
        self.assertEqual(count(cnn),count(shuffle))
        self.assertLess(abs(count(flat)/count(cnn)-1),.03)
        pooled=m.Network('pooled').eval();x=torch.randn(3,12,96)
        self.assertTrue(torch.allclose(pooled(x),pooled(x[:,torch.randperm(12)]),atol=1e-6))

    def test_float64_loss_curve_classifier(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(41)
        o=rng.normal(0,.1,(320,12,1));g=rng.normal(3,.1,(320,5,12,1))
        scores,c,diag=m.fit_origin(o,g,'loss_origin',51)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(diag['origin_auc'],.95)

    def test_matching_is_unique_and_respects_caliper(self):
        loss=np.array([1.,2.,3.,1.01,2.01,4.])
        p=np.array([1,1,1,0,0,0],bool);n=~p
        pairs=m.loss_match(loss,p,n,.02)
        self.assertEqual(len(pairs),2)
        self.assertEqual(len(np.unique(pairs)),4)
        self.assertTrue(np.all(abs(loss[pairs[:,0]]-loss[pairs[:,1]])<=.02))
        self.assertEqual(len(m.loss_match(loss,p,n,.001)),0)

    def test_probe_can_detect_independent_information(self):
        rng=np.random.default_rng(81)
        y=np.tile([0.,1.],128);x=rng.normal(size=(256,1))
        extra=y[:,None]+rng.normal(0,.1,(256,1))
        w,b=m.ridge_logistic(np.c_[x,extra],y,.001)
        ty=np.tile([0.,1.],128);tx=rng.normal(size=(256,1))
        te=ty[:,None]+rng.normal(0,.1,(256,1))
        p=m.sigmoid(np.c_[tx,te]@w+b)
        self.assertGreater(m.auc(p[ty==1],p[ty==0]),.99)
        self.assertLess(m.nll(p,ty).mean(),.1)


if __name__=='__main__':unittest.main()
