"""Independent hand checks + official reference checks; unittest works under -O."""
import math, tempfile, unittest
from pathlib import Path
import torch
from experiment import Block, Norm, sinusoidal, rotate_pairs, hand_ledger, run, setup
class Tests(unittest.TestCase):
    def test_manual_norm(self):
        x=torch.tensor([[[1.,2.,3.,4.],[7.,7.,7.,7.]]],dtype=torch.float64)
        y=Norm(4)(x)
        expected=torch.tensor([-1.5,-.5,.5,1.5],dtype=torch.float64)/math.sqrt(1.25+1e-5)
        torch.testing.assert_close(y[0,0],expected,rtol=0,atol=1e-14)
        torch.testing.assert_close(y[0,1],torch.zeros(4,dtype=torch.float64))
    def test_split_merge(self):
        x=torch.arange(24).reshape(1,3,8)
        split=x.reshape(1,3,2,4).transpose(1,2)
        self.assertEqual(split[0,1,2].tolist(),[20,21,22,23])
        self.assertTrue(torch.equal(split.transpose(1,2).reshape_as(x),x))
        self.assertFalse(torch.equal(split.reshape_as(x),x))
    def test_hand(self):
        r=hand_ledger();s=r[0]
        z=1/math.sqrt(1+1e-5);a=1/(1+math.exp(-2*math.sqrt(2)*z*z))
        self.assertAlmostEqual(s['a'][0][0][0],a,places=13)
        self.assertAlmostEqual(s['attention'][0][0][0],z*(1-2*a),places=13)
        self.assertGreater(r[0]['loss'],r[1]['loss']);self.assertGreater(r[1]['loss'],r[2]['loss'])
        for record in r:self.assertLess(record['manual_autograd_max'],2e-12)
        for i in [0,1]:
            for n,v in r[i]['parameters'].items():
                old=torch.tensor(v);grad=torch.tensor(r[i]['gradients'][n]);new=torch.tensor(r[i+1]['parameters'][n])
                torch.testing.assert_close(new,old-.05*grad,rtol=1e-6,atol=1e-7)
    def test_reference_and_invariants(self):
        with tempfile.TemporaryDirectory() as d: r=run(d)
        for row in r['alignment']:
            for key in ['output_max','input_gradient_max','parameter_gradient_max']:
                self.assertLess(row[key],2e-10)
        s=r['structure']
        for key in ['permutation_no_position_max','permutation_tokens_and_positions_max',
                    'causal_future_change_max','causal_prefix_extension_max','rope_joint_shift_max',
                    'rope_norm_max','causal_permuted_mask_max']:
            self.assertLess(s[key],2e-10)
        for key in ['permutation_fixed_positions_max','bidirectional_future_change_max',
                    'bidirectional_prefix_extension_max','causal_fixed_mask_max']:
            self.assertGreater(s[key],1e-5)
        self.assertEqual(s['parameter_count'],4*4*4+2*4*7+9*4+7)
    def test_different_lengths_and_masks(self):
        setup();m=Block()
        for length in [1,2,7]:
            x=torch.randn(3,length,4,dtype=torch.float64)
            allow=torch.ones(3,length,length,dtype=torch.bool).tril()
            self.assertEqual(m(x,allow).shape,x.shape)
        x=torch.randn(2,3,4,dtype=torch.float64)
        allow=torch.ones(2,3,3,dtype=torch.bool);allow[:,:,2]=False
        z=x.clone();z[:,2]+=torch.tensor([2.,0.,-3.,5.],dtype=torch.float64)
        # Only real queries compared; padding query output is not forced to zero.
        torch.testing.assert_close(m(x,allow)[:,:2],m(z,allow)[:,:2],rtol=0,atol=1e-13)
        with self.assertRaises(ValueError):m(x,torch.zeros(3,3,dtype=torch.bool))
        with self.assertRaises(ValueError):Block(d=5,heads=2)
    def test_rope_hand(self):
        q=torch.tensor([[1.,0.]],dtype=torch.float64)
        out=rotate_pairs(q,torch.tensor([math.pi/2]))
        torch.testing.assert_close(out,torch.tensor([[0.,1.]],dtype=torch.float64),atol=1e-7,rtol=0)
if __name__=='__main__':unittest.main(verbosity=2)
