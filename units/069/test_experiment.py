import unittest,tempfile,json
from pathlib import Path
import torch
from experiment import run,verify_assets,digest
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=tempfile.TemporaryDirectory();cls.r=run(cls.d.name);cls.p=Path(cls.d.name)
    @classmethod
    def tearDownClass(cls):cls.d.cleanup()
    def test_full_resume(self):
        self.assertEqual(self.r['resume_max_parameter_gap'],0);self.assertEqual(self.r['resume_loss_max_gap'],0)
    def test_wrong_resume(self):self.assertGreater(self.r['weights_only_max_parameter_gap'],1e-5)
    def test_integrity_and_tamper(self):
        manifest=json.loads((self.p/'asset_manifest.json').read_text());self.assertEqual(verify_assets(self.p,manifest),[])
        self.assertEqual(verify_assets(self.p,{'config.json':'0'*64}),['config.json'])
        self.assertIn('outside_root',verify_assets(self.p,{'../secret':'x'})[0])
        self.assertTrue(self.r['simulated_tamper_detected'])
    def test_safe_checkpoint_and_splits(self):
        c=torch.load(self.p/'resume_epoch12.pt',weights_only=True);self.assertEqual(c['epoch'],12)
        self.assertIn('optimizer',c);self.assertIn('torch_rng',c);self.assertIn('order_rng',c)
        self.assertEqual(self.r['duplicate_demo']['duplicates'],4)
if __name__=='__main__':unittest.main()
