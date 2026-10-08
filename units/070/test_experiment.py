import unittest,torch,numpy as np
from experiment import model,data,paired,SEEDS,MODES
class Tests(unittest.TestCase):
    def test_split(self):
        x,y=data();self.assertEqual(x.shape,(832,20));self.assertEqual(y.shape,(832,));self.assertEqual(len(set(range(128))&set(range(128,320))),0)
    def test_matched_initialization(self):
        torch.manual_seed(11);a=model(0.).state_dict();torch.manual_seed(11);b=model(.4).state_dict()
        for k in a:torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
    def test_paired_interval(self):
        r=paired([1,2,3,4,5]);self.assertEqual(r['mean'],3);self.assertLess(r['ci95'][0],3);self.assertGreater(r['ci95'][1],3)
        self.assertEqual(paired([2]*5)['ci95'],[2,2])
    def test_inverted_dropout(self):
        torch.manual_seed(70);d=torch.nn.Dropout(.5);z=d(torch.ones(100000))
        self.assertLess(abs(float(z.mean())-1),.02);d.eval();torch.testing.assert_close(d(torch.ones(4)),torch.ones(4),atol=0,rtol=0)
    def test_budget(self):self.assertEqual(len(SEEDS)*len(MODES)*160,3200)
if __name__=='__main__':unittest.main()
