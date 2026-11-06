import unittest
import numpy as np
from membrain_word_probe import score_record


class WordProbeTests(unittest.TestCase):
    def test_text_event_frequency_and_fixed_smoothing(self):
        q=dict(prefix='A prefix',suffix='target ending.',progress=.2,answer=dict(greedy='target.',samples=['target x']*8+['other x']*8))
        r=dict(text='A prefix target ending.',queries=[q,dict(q,progress=.6)])
        s,c=score_record(r)
        self.assertAlmostEqual(s['Membrain next-word probe'],np.log(.5))
        self.assertEqual(s['Greedy next-word recovery'],1)
        self.assertEqual(s['Sampled next-word recovery'],.5)


if __name__=='__main__':unittest.main()
