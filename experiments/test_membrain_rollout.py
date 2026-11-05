import unittest
import numpy as np
import torch
from torch.nn import functional as F
import membrain_rollout as m


class RolloutChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.lm=m.prev.base.LM();self.docs=torch.randint(1,53,(3,48))

    def test_scores_true_targets_on_generated_history(self):
        # Force generated tokens to zero, which never occur in these true suffixes.
        sampler=lambda probs:torch.zeros(len(probs),1,dtype=torch.long)
        values=m.free_running_losses(self.lm,self.docs,8,2,sampler=sampler)
        prefix=self.docs[:,:8].repeat_interleave(2,0)
        for j in range(8):
            history=torch.cat([prefix,torch.zeros(len(prefix),j,dtype=torch.long)],1)
            with torch.no_grad():
                logits=self.lm(history)[:,-1]
                expected=F.cross_entropy(logits,self.docs[:,8+j].repeat_interleave(2),reduction='none')
            self.assertTrue(torch.allclose(values[:,:,j].reshape(-1),expected,atol=1e-6))

    def test_true_tokens_do_not_enter_generation_history(self):
        sampler=lambda probs:torch.zeros(len(probs),1,dtype=torch.long)
        original=m.free_running_losses(self.lm,self.docs,8,1,sampler=sampler)
        altered=self.docs.clone();altered[:,8]=(altered[:,8]+1)%53
        changed=m.free_running_losses(self.lm,altered,8,1,sampler=sampler)
        self.assertFalse(torch.allclose(original[:,:,0],changed[:,:,0]))
        self.assertTrue(torch.equal(original[:,:,1:],changed[:,:,1:]))

    def test_first_token_matches_teacher_forcing_and_uniform_model(self):
        o,g,t,error=m.curves(self.lm,self.docs,2)
        self.assertLess(error,1e-5)
        for p in self.lm.parameters():p.data.zero_()
        o,g,t,error=m.curves(self.lm,self.docs,2)
        self.assertTrue(np.allclose(g,np.log(53)))
        self.assertTrue(np.allclose(o,g.mean(1)))


if __name__=='__main__':unittest.main()
