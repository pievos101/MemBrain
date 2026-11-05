import unittest
import numpy as np
import torch
import membrain_recovery as m


class RecoveryChecks(unittest.TestCase):
    def test_sentence_pool_and_rotation(self):
        texts,background=m.sentence_pool()
        self.assertEqual(len(set(texts+background)),256)
        self.assertTrue(all(80<=len(t)<=120 for t in texts))
        groups=np.repeat(np.arange(4),32)
        exposures=np.stack([m.LEVELS[(groups+r)%4] for r in range(4)])
        self.assertTrue(np.all(np.sort(exposures,axis=0)==m.LEVELS[:,None]))

    def test_threshold_is_sustained_and_censoring_preserved(self):
        p=np.zeros((2,len(m.FRACTIONS)))
        p[0,1:]=.8;p[0,3]=.4
        result=m.sustained_threshold(p)
        self.assertEqual(result[0],m.FRACTIONS[4])
        self.assertTrue(np.isinf(result[1]))

    def test_packed_prefix_and_greedy_reconstruction(self):
        torch.set_num_threads(2);torch.manual_seed(4)
        lm=m.SentenceLM()
        texts=['A'*80,'B'*100]
        # Make the decoder always emit A, independent of prefix and history.
        for p in lm.parameters():p.data.zero_()
        lm.output.bias.data[ord('A')-31]=50
        values,generated=m.recover(lm,texts,2)
        self.assertTrue(np.all(values['suffix_exact'][0,:,0]==1))
        self.assertTrue(np.all(values['tail_exact'][0,:,0]==1))
        self.assertTrue(np.all(values['suffix_accuracy'][1,:,0]==0))
        for j in range(len(m.FRACTIONS)):
            for i in range(2):
                self.assertEqual(len(generated[j][i]),len(texts[i])-values['prefix_sizes'][i,j])
        self.assertTrue(np.all(values['lengths'][:,None]-values['prefix_sizes']>=8))

    def test_greedy_matches_individual_unpadded_model(self):
        torch.set_num_threads(2);torch.manual_seed(19)
        lm=m.SentenceLM().eval();texts=['A sentence with assorted characters. '*3,'Another example. '*5]
        texts=[texts[0][:100],texts[1][:80]]
        values,generated=m.recover(lm,texts,0)
        for j in [0,4,11]:
            for i,t in enumerate(texts):
                n=int(values['prefix_sizes'][i,j]);history=[ord(c)-31 for c in t[:n]];prediction=[]
                for _ in range(3):
                    with torch.no_grad():logits=lm(torch.tensor([history]))[0,-1]
                    token=int(logits[1:].argmax())+1
                    history.append(token);prediction.append(chr(token+31))
                self.assertEqual(generated[j][i][:3],''.join(prediction))

    def test_true_suffix_does_not_change_generated_path(self):
        torch.set_num_threads(2);torch.manual_seed(23)
        lm=m.SentenceLM().eval();texts=['A'*80,'A'*8+'B'*72]
        values,generated=m.recover(lm,texts,0)
        for j in range(4):
            self.assertEqual(generated[j][0],generated[j][1])


if __name__=='__main__':unittest.main()
