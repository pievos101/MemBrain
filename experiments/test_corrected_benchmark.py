"""Run: python -m unittest discover -s experiments -p 'test_corrected_benchmark.py'."""
import unittest
import numpy as np
import torch
import corrected_benchmark as b


class BenchmarkChecks(unittest.TestCase):
    def test_disjoint_reproducible_corpus(self):
        x, groups = b.corpus(1)
        self.assertTrue(torch.equal(x, b.corpus(1)[0]))
        self.assertFalse(torch.equal(x, b.corpus(2)[0]))
        self.assertEqual(len({tuple(row.tolist()) for row in x}), 340)
        self.assertEqual(sum(len(v) for v in groups.values()), 340)

    def test_auc_direction_and_ties(self):
        self.assertEqual(b.auc(np.array([2, 3]), np.array([0, 1])), 1.)
        self.assertEqual(b.auc(np.array([0, 1]), np.array([2, 3])), 0.)
        self.assertEqual(b.auc(np.ones(3), np.ones(4)), .5)

    def test_horizon_and_order_invariant_control(self):
        torch.set_num_threads(2)
        x, _ = b.corpus(1)
        model = b.LM()
        o, g, loss = b.features(model, x[:3], 3)
        self.assertEqual(tuple(g.shape), (3, 3, 4, 48))
        self.assertTrue(np.isfinite(loss).all())
        self.assertTrue(torch.isfinite(g).all())
        with torch.no_grad():
            h, _ = model.gru(model.embedding(x[:3, :16]))
            self.assertTrue(torch.allclose(o[:, 0], h[:, 8:16].mean(1)))
            attack = b.Attack(False).eval()
            self.assertTrue(torch.allclose(attack(o), attack(o[:, [2, 0, 3, 1]]), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
