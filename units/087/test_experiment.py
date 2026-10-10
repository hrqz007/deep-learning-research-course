import unittest,numpy as np
import experiment as e
class Tests(unittest.TestCase):
 def setUp(self):self.docs,self.tasks=e.fixture()
 def test_unique_ids(self):self.assertEqual(len({t['id'] for t in self.tasks}),80)
 def test_groups(self):self.assertEqual({g:sum(t['group']==g for t in self.tasks) for g in ['exact','alias','missing','denied']},{g:20 for g in ['exact','alias','missing','denied']})
 def test_no_gold_input(self):
  t=self.tasks[0];r=e.run_system({'id':t['id'],'query':t['query']},self.docs);self.assertTrue(e.score(t,r)['success'])
 def test_alias(self):
  t=self.tasks[20];self.assertEqual(e.run_system(t,self.docs)['citation'],'E020')
 def test_ablation(self):
  t=self.tasks[20];self.assertFalse(e.score(t,e.run_system(t,self.docs,'no_normalization'))['success'])
 def test_denied(self):self.assertEqual(e.run_system(self.tasks[60],self.docs)['status'],'denied')
 def test_missing(self):self.assertEqual(e.run_system(self.tasks[40],self.docs)['status'],'abstain')
 def test_citation_required(self):
  t=self.tasks[0];s=e.score(t,e.run_system(t,self.docs,'no_citation'));self.assertEqual(s['value_only'],1);self.assertEqual(s['success'],0)
 def test_budget(self):
  for t in self.tasks:self.assertLessEqual(e.run_system(t,self.docs,budget=1)['attempts'],1)
 def test_bootstrap_identity(self):self.assertEqual(e.paired_bootstrap([1,0],[1,0])['ci95'],[0.,0.])
 def test_bootstrap_pairing(self):self.assertEqual(e.paired_bootstrap([1,1],[0,0])['mean_difference'],1)
 def test_bootstrap_shape(self):
  with self.assertRaises(ValueError):e.paired_bootstrap([1],[0,1])
 def test_repeat(self):self.assertEqual(e.run_system(self.tasks[0],self.docs),e.run_system(self.tasks[0],self.docs))
 def test_hash_mutation(self):
  h=e.digest(self.tasks);self.tasks[0]['query']='changed';self.assertNotEqual(h,e.digest(self.tasks))
if __name__=='__main__':unittest.main()
