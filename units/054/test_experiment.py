"""Data and gradient checks remain active with python -O."""
import unittest,math,copy,tempfile
import torch
from experiment import *
from generate_data import normalize
class Tests(unittest.TestCase):
    def test_shift_boundary(self):
        b=ar_packed([[4,7],[5]])
        self.assertEqual(b['x'].tolist(),[[1,4,7,1,5]])
        self.assertEqual(b['y'].tolist(),[[4,7,2,5,2]])
        self.assertEqual(b['position'].tolist(),[[0,1,2,0,1]])
        self.assertFalse(b['allow'][0,3,2]);self.assertTrue(b['allow'][0,4,3])
        self.assertFalse(b['allow'][0,0,1])
    def test_mlm(self):
        b=mlm_padded([[4,7],[5]],rate=1)
        self.assertEqual(b['x'].tolist(),[[BOS,MASK,MASK,EOS],[BOS,MASK,EOS,PAD]])
        self.assertEqual(b['y'].tolist(),[[IGNORE,4,7,IGNORE],[IGNORE,5,IGNORE,IGNORE]])
        self.assertEqual(int((b['y']!=IGNORE).sum()),3)
        with self.assertRaises(ValueError):mlm_padded([[4]],rate=0)
    def test_loss_and_gradient(self):
        logits=torch.tensor([[[math.log(2),0.,0.],[0.,math.log(3),0.],[9.,-4.,1.]]],dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([[0,1,IGNORE]])
        total,count,mean=masked_nll(logits,labels)
        self.assertEqual(count,2);self.assertAlmostEqual(total.item(),-math.log(.5)-math.log(.6),places=13)
        mean.backward();expected=torch.tensor([[[-.25,.125,.125],[.1,-.2,.1],[0.,0.,0.]]],dtype=torch.float64)
        torch.testing.assert_close(logits.grad,expected,atol=1e-14,rtol=0)
        changed=logits.detach().clone();changed[0,2]=torch.tensor([1e3,-1e3,0.])
        self.assertAlmostEqual(masked_nll(changed,labels)[2].item(),mean.item(),places=13)
        with self.assertRaises(ValueError):masked_nll(logits,torch.full_like(labels,IGNORE))
    def test_data_split(self):
        rows,duplicates=corpus();self.assertEqual(len(rows),10);self.assertEqual(len(duplicates),1)
        sets={s:{r['normalized_sha256'] for r in rows if r['split']==s} for s in ['train','validation','test']}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:self.assertFalse(sets[a]&sets[b])
        self.assertEqual(normalize('蓝  猫\n追 红 球'),'蓝 猫 追 红 球')
    def test_packing_gradients(self):
        setup();docs=[[4,7],[5],[6,9,10,13]];a=ar_padded(docs);b=ar_packed(docs)
        m=TinyProbe();n=copy.deepcopy(m)
        la=m(a);lb=n(b)
        torch.testing.assert_close(la[a['valid']],lb[0],atol=2e-12,rtol=0)
        masked_nll(la,a['y'])[2].backward();masked_nll(lb,b['y'])[2].backward()
        for (name,p),(name2,q) in zip(m.named_parameters(),n.named_parameters()):
            self.assertEqual(name,name2);torch.testing.assert_close(p.grad,q.grad,atol=2e-12,rtol=0)
    def test_ignored_context_can_receive_gradient(self):
        setup();b=mlm_padded([[4,7,10]],rate=1)
        b['x'][0]=torch.tensor([BOS,4,MASK,10,EOS]);b['y'][0]=torch.tensor([IGNORE,IGNORE,7,IGNORE,IGNORE])
        m=TinyProbe();logits=m(b);logits.retain_grad();masked_nll(logits,b['y'])[2].backward()
        self.assertEqual(logits.grad[0,1].abs().max().item(),0.)
        self.assertGreater(m.embed.weight.grad[4].abs().max().item(),1e-8)
    def test_scope_and_failures(self):
        with self.assertRaises(ValueError):ar_packed([])
        with self.assertRaises(ValueError):ar_padded([[BOS,4]])
        with self.assertRaises(ValueError):ar_padded([[]])
        with tempfile.TemporaryDirectory() as d:r=run(d)
        self.assertLess(r['packing_max_logit_error'],2e-12)
        self.assertEqual(r['blocked_cross_document_change'],0.)
        self.assertGreater(r['leaky_cross_document_change'],.01)
        self.assertGreater(r['missing_position_reset_max_error'],.1)
        self.assertLess(r['copy_diagnostic']['wrong_unshifted_nll'],.001)
        self.assertGreater(r['copy_diagnostic']['correct_shifted_nll'],10.)
        self.assertEqual(r['ar']['count'],34);self.assertEqual(r['mlm']['count'],10)
if __name__=='__main__':unittest.main(verbosity=2)
