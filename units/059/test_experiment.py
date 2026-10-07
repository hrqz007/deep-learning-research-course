"""Explicit unittest checks remain active under Python -O."""
from pathlib import Path
import unittest,json,math
import numpy as np
import torch
from experiment import *
class Checks(unittest.TestCase):
    def test_nce_hand_and_masks(self):
        a=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True);b=a.detach().clone().requires_grad_()
        value=info_nce(a,b,1.);self.assertAlmostEqual(value.item(),math.log(math.e+2)-1,6)
        value.backward();self.assertTrue(torch.isfinite(a.grad).all())
        collapsed=torch.ones((3,2));self.assertAlmostEqual(info_nce(collapsed,collapsed).item(),math.log(5),6)
        self.assertAlmostEqual(info_nce(a.detach(),b.detach()).item(),info_nce(b.detach(),a.detach()).item(),7)
    def test_guards_and_augmentation(self):
        for t in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):info_nce(torch.ones(2,2),torch.ones(2,2),t)
        with self.assertRaises(ValueError):info_nce(torch.ones(1,2),torch.ones(1,2))
        x=torch.from_numpy(make_data()['x_train'][:128]);v=augment(x,'good',torch.Generator().manual_seed(1))
        self.assertLess((v[:,:4]-x[:,:4]).square().mean(),.01)
        self.assertGreater((v[:,4:]-x[:,4:]).square().mean(),1.)
    def test_probe_and_replay(self):
        out=OUTPUT;r=json.loads((out/'results.json').read_text());self.assertEqual(len(r['results']),12)
        expected={(m,s) for m in METHODS for s in SEEDS};self.assertEqual({(v['method'],v['seed']) for v in r['results']},expected)
        self.assertEqual(len(list(out.glob('weights_*.npz'))),12);self.assertEqual(len(list(out.glob('probe_*.npz'))),12)
        d=np.load(out/'data.npz');plot=np.load(out/'plot_data.npz')
        for k,v in make_data().items():np.testing.assert_array_equal(d[k],v)
        for row in r['results']:
            key=f"{row['method']}_{row['seed']}";m=load_model(out/f'weights_{key}.npz');before={k:v.clone() for k,v in m.state_dict().items()}
            with torch.no_grad():z=m(torch.from_numpy(d['x_test'])).numpy();zt=m(torch.from_numpy(d['x_train'])).numpy()
            p=fit_probe(zt[:128],d['y_train'][:128]);saved=np.load(out/f'probe_{key}.npz')
            for k,v in p.items():np.testing.assert_allclose(v,saved[k],atol=1e-7)
            np.testing.assert_allclose(z,plot[key+'_embedding'],atol=1e-6)
            score=probe_scores(z,p);np.testing.assert_allclose(score,plot[key+'_scores'],atol=1e-6)
            self.assertAlmostEqual(float((score.argmax(1)==d['y_test']).mean()),row['test_accuracy'],10)
            self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))
        p=fit_probe(np.ones((8,2)),np.tile(np.arange(4),2));self.assertTrue(np.isfinite(p['weight']).all())
def main(output=None):
    global OUTPUT;OUTPUT=Path(output) if output else Path(__file__).resolve().parent/'outputs';torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():raise RuntimeError('DL059 checks failed')
if __name__=='__main__':main()
