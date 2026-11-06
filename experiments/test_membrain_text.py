import ast
import inspect
import unittest
from pathlib import Path
import numpy as np
import torch
import membrain_text as method
from membrain_text_backend import TextCompletionService


class TextTests(unittest.TestCase):
    def test_distances_use_only_normalized_text(self):
        s=method.similarity('One two three four.','ONE two nine four!')
        self.assertEqual(s['word_edit'],.75);self.assertEqual(s['rouge_l'],.75)
        self.assertEqual(method.similarity('hello','')['word_edit'],0)
        self.assertEqual(method.similarity('A b c d e','x b c d e',tail=4)['exact'],1)

    def test_query_boundary_preserves_fixed_tail_and_suffix_never_enters_callback(self):
        text='Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda omega.'
        calls=[]
        def completion(prefixes,**settings):
            calls.extend(prefixes)
            self.assertEqual(settings['samples'],3)
            for p in prefixes:self.assertLessEqual(len(method.words(p)),len(method.words(text))-4)
            return [dict(greedy='',samples=['']*3) for _ in prefixes]
        records=method.collect([text],completion);scores,curves=method.score_record(records[0])
        self.assertTrue(calls);self.assertEqual(scores['Membrain text'],0)
        self.assertTrue(np.all(np.diff(curves['progress'])>0))

    def test_core_has_no_model_or_tokenizer_import_and_no_label_argument(self):
        tree=ast.parse(Path(method.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any('torch' in s or 'tokenizer' in s or 'backend' in s for s in imports))
        self.assertEqual(list(inspect.signature(method.fit_mixture).parameters),['scores'])

    def test_bic_abstains_on_identical_scores(self):
        m=method.fit_mixture(np.full(128,.2));self.assertFalse(m['selected_two'])

    def test_mixture_orientation_is_high_recovery_without_labels(self):
        rng=np.random.default_rng(8);x=np.r_[rng.normal(.1,.015,90),rng.normal(.8,.02,38)]
        m=method.fit_mixture(x);self.assertTrue(m['selected_two']);self.assertLess(m['means'][0],m['means'][1])
        p=method.component_probability(m,[.1,.8]);self.assertLess(p[0],.01);self.assertGreater(p[1],.99)

    def test_backend_compacts_finished_rows_and_returns_text_only(self):
        class Tokenizer:
            def encode(self,text):
                return type('Encoded',(),{'ids':[9] if text=='\n' else [3 if text=='a' else 4]})()
            def decode(self,ids):return ''.join({5:'c'}.get(i,'?') for i in ids)
        class Model(torch.nn.Module):
            max_length=20
            def forward(self,ids,cache=None,return_hidden=False):
                n,t=ids.shape;logits=torch.full((n,t,10),-10000.);next_token=torch.where(ids[:,-1]==4,5,9)
                logits[torch.arange(n),-1,next_token]=0
                k=ids[:,None,:,None].float()
                if cache is not None:k=torch.cat([cache[0][0],k],dim=2)
                return logits,torch.zeros(n,t,1),[(k,k)]
        service=TextCompletionService(Model(),Tokenizer())
        out=service(['a','b'],samples=3,temperature=.8,max_new_tokens=4,seed=3)
        self.assertEqual(out,[dict(greedy='',samples=['']*3),dict(greedy='c',samples=['c']*3)])


if __name__=='__main__':unittest.main()
