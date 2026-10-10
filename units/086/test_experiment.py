import unittest,copy
import experiment as e
class Tests(unittest.TestCase):
 def setUp(self):self.task={'id':'test','query':'experiment E03 yield','target':'E03'}
 def test_retrieval(self):self.assertEqual(e.retrieve('E03',e.corpus())[0]['id'],'E03')
 def test_success(self):self.assertEqual(e.run_task(self.task)['answer']['value'],1.625)
 def test_retry(self):self.assertEqual(e.run_task(self.task,fail_once=True)['tool_attempts'],2)
 def test_exhaustion(self):self.assertEqual(e.run_task(self.task,fail_once=True,max_attempts=1)['events'][-1]['state'],'EXHAUSTED')
 def test_missing(self):self.assertIsNone(e.run_task({**self.task,'target':'E99'})['answer'])
 def test_tool_denial(self):
  with self.assertRaises(PermissionError):e.Sandbox().call({'id':'x','tool':'delete_all','args':{}})
 def test_schema(self):
  with self.assertRaises(ValueError):e.Sandbox().call({'id':'x','tool':'read_record','args':{'record_id':'E03','shell':'rm'}})
 def test_capability(self):
  with self.assertRaises(PermissionError):e.Sandbox().call({'id':'x','tool':'read_record','args':{'record_id':'DANGER'}})
 def test_idempotence(self):
  b=e.Sandbox();r={'id':'n','tool':'save_note','args':{'text':'one'}};self.assertEqual(b.call(r),b.call(r));self.assertEqual(len(b.ledger),1);self.assertEqual(b.calls,1)
 def test_key_conflict(self):
  b=e.Sandbox();b.call({'id':'n','tool':'save_note','args':{'text':'one'}})
  with self.assertRaises(ValueError):b.call({'id':'n','tool':'save_note','args':{'text':'two'}})
 def test_replay(self):self.assertTrue(e.replay(e.run_task(self.task))['verified'])
 def test_tamper(self):
  t=e.run_task(self.task);t['answer']['value']=10
  with self.assertRaises(ValueError):e.replay(t)
 def test_corpus_change(self):
  t=e.run_task(self.task);docs=e.corpus();docs[0]['value']=0
  with self.assertRaises(ValueError):e.replay(t,docs)
 def test_sum(self):self.assertEqual(e.Sandbox().call({'id':'s','tool':'sum_values','args':{'record_ids':['E00','E01']}})['value'],2.625)
if __name__=='__main__':unittest.main()
