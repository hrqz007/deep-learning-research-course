"""Fast semantic checks; unittest checks remain active under python -O."""
import tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from experiment import Model, data, predict, timing
class Tests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1);torch.manual_seed(68)
    def test_split_and_preprocessing(self):
        x,y=data();self.assertEqual(x.shape,(1152,4));self.assertEqual(y.shape,(1152,))
        m=Model(x[:768].mean(0),x[:768].std(0));z=(torch.tensor(x[:768])-m.mean)/m.scale
        np.testing.assert_allclose(z.mean(0),0,atol=1e-5)
    def test_export_and_guard(self):
        m=Model(np.zeros(4),np.ones(4)).eval();x=torch.randn(1,4)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.pt2';torch.export.save(torch.export.export(m,(x,),strict=True),path)
            loaded=torch.export.load(path).module()
            with torch.inference_mode():torch.testing.assert_close(m(x),loaded(x),rtol=1e-5,atol=1e-6)
            with self.assertRaises((RuntimeError,AssertionError,ValueError)):loaded(torch.randn(2,4))
    def test_request_rejection(self):
        m=Model(np.zeros(4),np.ones(4)).eval()
        self.assertEqual(predict(m,np.zeros((1,4),dtype='float32'))['status'],'ok')
        for x in [np.zeros((4,),dtype='float32'),np.full((1,4),np.inf,dtype='float32'),np.zeros((1,4))]:
            self.assertIsNone(predict(m,x)['prediction'])
    def test_eval_and_inference_are_independent(self):
        m=Model(np.zeros(4),np.ones(4)).train();x=torch.randn(32,4)
        with torch.inference_mode(): a=m(x);b=m(x)
        self.assertGreater(float((a-b).abs().max()),0)
        m.eval()
        with torch.inference_mode():torch.testing.assert_close(m(x),m(x),atol=0,rtol=0)
    def test_backend_fallback(self):
        class Broken(torch.nn.Module):
            def forward(self,x):raise RuntimeError('simulated backend failure')
        self.assertEqual(predict(Broken(),np.zeros((1,4),dtype='float32'))['status'],'unavailable')
        self.assertEqual(predict(Broken(),[[1],[1,2]])['status'],'rejected')
    def test_quantile_definition(self):
        self.assertAlmostEqual(float(np.quantile([1,2,3,4,100],.95,method='linear')),80.8)
if __name__=='__main__':unittest.main()
