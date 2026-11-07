import inspect
import unittest
import numpy as np
from membrain_curve_cluster import GRID,fit,predict,early_cost,feature_sets


class CurveClusterTests(unittest.TestCase):
    def test_fitter_uses_shapes_that_scalar_summaries_cannot_distinguish(self):
        rng=np.random.default_rng(4);w=early_cost(np.eye(8));basis=np.stack([np.ones(8),w],axis=1)
        v=rng.normal(size=8);v-=basis@np.linalg.lstsq(basis,v,rcond=None)[0];v*=.3/np.max(np.abs(v))
        noise=rng.normal(0,.004,(80,8));noise-=noise@basis@np.linalg.inv(basis.T@basis)@basis.T
        x=np.r_[np.tile(.5+v,(40,1)),np.tile(.5-v,(40,1))]+noise
        np.testing.assert_allclose(early_cost(x),.5,atol=1e-12)
        np.testing.assert_allclose(x.mean(1),.5,atol=1e-12)
        model=fit(x);self.assertEqual(model['components'],2)
        p,_=predict(model,x);self.assertGreater(abs(p[:40].mean()-p[40:].mean()),.9)
        self.assertEqual(fit(x.mean(1,keepdims=True))['components'],1)

    def test_constant_curves_abstain_and_do_not_force_two_classes(self):
        model=fit(np.full((64,8),.8));self.assertEqual(model['components'],1)
        p,labels=predict(model,np.full((3,8),.8))
        np.testing.assert_array_equal(p,.5);np.testing.assert_array_equal(labels,-1)

    def test_posterior_independent_of_other_test_candidates_and_no_label_argument(self):
        rng=np.random.default_rng(1);x=np.r_[rng.normal(.15,.02,(40,8)),rng.normal(.8,.02,(40,8))]
        m=fit(x);p,_=predict(m,x);one,_=predict(m,x[:1]);self.assertEqual(one[0],p[0])
        self.assertEqual(list(inspect.signature(fit).parameters),['x','max_components'])

    def test_text_features_retain_all_prefix_coordinates(self):
        text='One two three four five six seven eight nine ten eleven twelve.'
        queries=[]
        for i,t in enumerate(GRID):
            queries.append(dict(prefix=str(i),suffix='nine ten eleven twelve',progress=t*(1-4/12),
                answer=dict(greedy='nine ten eleven twelve',samples=['nine x']*16)))
        short=dict(text=text,queries=queries);long=dict(text=text,queries=queries)
        f=feature_sets([short],[long]);self.assertEqual(f['Whole curve'].shape,(1,8));self.assertEqual(f['Joint curves'].shape,(1,16))
        np.testing.assert_array_equal(f['Whole curve'],0);np.testing.assert_array_equal(f['Next-word curve'],0)


if __name__=='__main__':unittest.main()
