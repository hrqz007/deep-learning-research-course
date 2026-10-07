from pathlib import Path
import unittest,json,math
import numpy as np
import torch
from experiment import *
class Checks(unittest.TestCase):
    def test_kl_and_reparameterization(self):
        mu=torch.tensor([[1.,0.]],dtype=torch.float64,requires_grad=True);lv=torch.tensor([[math.log(4),0.]],dtype=torch.float64,requires_grad=True)
        self.assertAlmostEqual(kl_standard(mu,lv).item(),.5*(4-math.log(4)),12)
        eps=torch.tensor([[.5,-1.]],dtype=torch.float64);z=reparameterize(mu,lv,eps)
        np.testing.assert_allclose(z.detach().numpy(),[[2.,-1.]])
        z.sum().backward();np.testing.assert_allclose(mu.grad.numpy(),[[1.,1.]]);np.testing.assert_allclose(lv.grad.numpy(),[[.5,-.5]])
        self.assertAlmostEqual(kl_standard(torch.zeros(1,2),torch.zeros(1,2)).item(),0.)
    def test_nll_reductions_guards(self):
        z=torch.zeros(2,2,dtype=torch.float64);self.assertAlmostEqual(reconstruction_nll(z,z,'xy',1).mean().item(),math.log(2*math.pi),12)
        self.assertAlmostEqual(reconstruction_nll(z,z,'image').mean().item(),2*math.log(2),12)
        with self.assertRaises(ValueError):reconstruction_nll(z,z,'xy',0)
        with self.assertRaises(ValueError):reconstruction_nll(z,torch.ones_like(z)*2,'image')
        with self.assertRaises(ValueError):kl_standard(torch.zeros(0,2),torch.zeros(0,2))
        with self.assertRaises(ValueError):reparameterize(z,z,torch.zeros(1,2))
    def test_saved_full_replay(self):
        out=OUTPUT;d=np.load(out/'data.npz');plots=np.load(out/'plot_data.npz');r=json.loads((out/'results.json').read_text())
        expected={(kind,config,seed) for kind in ['xy','image'] for config in CONFIGS for seed in SEEDS}
        self.assertEqual(len(r['results']),12);self.assertEqual({(x['kind'],x['config'],x['seed']) for x in r['results']},expected);self.assertEqual(len(list(out.glob('weights_*.npz'))),12)
        for k,v in make_data().items():np.testing.assert_array_equal(d[k],v)
        for row in r['results']:
            kind=row['kind'];key=f"{kind}_{row['config']}_{row['seed']}";m=load_model(out/f'weights_{key}.npz',2 if kind=='xy' else 64)
            metrics,a=evaluate(m,torch.from_numpy(d[kind+'_test']),kind)
            for k,v in metrics.items():self.assertAlmostEqual(v,row[k],5)
            for k,v in a.items():np.testing.assert_allclose(v,plots[key+'_'+k],atol=2e-5)
            self.assertAlmostEqual(row['negative_elbo'],row['mc_nll']+row['kl'],4)
            self.assertTrue(np.isfinite(plots[key+'_samples']).all())
            if kind=='image':self.assertTrue(np.isin(plots[key+'_samples'],[0,1]).all())
        for kind in ['xy','image']:
            high=[x for x in r['results'] if x['kind']==kind and x['config']=='high_beta'];rest=[x for x in r['results'] if x['kind']==kind and x['config']=='restore_beta']
            self.assertTrue(all(x['kl']<.001 for x in high));self.assertTrue(all(x['kl']>1 for x in rest))
        for kind in ['xy','image']:
            for seed in SEEDS:
                np.testing.assert_array_equal(plots[f'{kind}_high_beta_{seed}_trace'][:30],plots[f'{kind}_restore_beta_{seed}_trace'][:30])
def main(output=None):
    global OUTPUT;OUTPUT=Path(output) if output else Path(__file__).resolve().parent/'outputs';torch.set_num_threads(1)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not r.wasSuccessful():raise RuntimeError('DL060 checks failed')
if __name__=='__main__':main()
