import unittest
import numpy as np
from membrain_text_validation import collect,combined_scores,rank_correlation
from membrain_word_probe import score_record


class ValidationTests(unittest.TestCase):
    def test_primary_score_preserved_and_budgets_nested(self):
        text='Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda omega.'
        def completion(prefixes,**settings):
            n=settings['samples'];return [dict(greedy='wrong ending',samples=['wrong ending']*n) for p in prefixes]
        a=collect([text],completion,16,8,1);b=collect([text],completion,16,8,2);c=collect([text],completion,0,96,3)
        got=combined_scores(a,b,c);prior,_=score_record(a[0])
        self.assertEqual(got['Membrain 16'][0],prior['Membrain next-word probe'])
        self.assertAlmostEqual(got['Membrain nested3'][0],np.log(.5/4))
        self.assertAlmostEqual(got['Membrain pooled32'][0],np.log(.5/33))

    def test_replay_is_candidate_independent_and_no_suffix_sent(self):
        text='Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda omega.'
        def completion(prefixes,**settings):
            for p in prefixes:self.assertLessEqual(len(p.split()),8)
            return [dict(greedy='',samples=['']*settings['samples']) for p in prefixes]
        a=collect([text,text],completion,16,8,1);c=collect([text,text],completion,0,96,1)
        all_scores=combined_scores(a,a,c);single=combined_scores(a[:1],a[:1],c[:1])
        for key in all_scores:np.testing.assert_allclose(single[key],all_scores[key][:1])

    def test_tie_aware_query_stability(self):
        self.assertAlmostEqual(rank_correlation([1,1,2,3],[1,1,2,3]),1)
        self.assertAlmostEqual(rank_correlation([1,2,3],[3,2,1]),-1)
        self.assertIsNone(rank_correlation([1,1],[2,3]))


if __name__=='__main__':unittest.main()
