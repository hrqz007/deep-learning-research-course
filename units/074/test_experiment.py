"""用可手算总体和有限类检查统计实现；普通与-O模式均执行。"""
import unittest
import numpy as np
from generate_data import population,sample
from experiment import radius,true_risks,empirical_risks,memorize

class Tests(unittest.TestCase):
    def test_radius(self):self.assertAlmostEqual(radius(100,23),np.sqrt(np.log(920)/200))
    def test_monotonic_n(self):self.assertLess(radius(200,23),radius(100,23))
    def test_monotonic_class(self):self.assertGreater(radius(100,46),radius(100,23))
    def test_bad_inputs(self):
        for args in [(0,23,.05),(100,0,.05),(100,23,1),(100,23,0),(1.5,2,.1)]:
            with self.assertRaises(ValueError):radius(*args)
    def test_constant_risk(self):
        p=np.array([.2,.8]);np.testing.assert_allclose(true_risks(np.array([[0,0],[1,1],[0,1]]),p),[.5,.5,.2])
    def test_empirical_hand(self):
        v=empirical_risks(np.array([[0,1],[1,0]]),np.array([0,1,1]),np.array([0,0,1]));np.testing.assert_allclose(v,[1/3,2/3])
    def test_population_best(self):
        p=population();self.assertAlmostEqual(true_risks(p['predictions'],p['prob']).min(),.15)
    def test_fixed_class(self):
        p=population();self.assertEqual(p['predictions'].shape,(23,21));self.assertEqual(len(np.unique(p['predictions'],axis=0)),22)
        # 23个命名阈值有一对给出相同离散标签表；用23作上界仍正确。
    def test_memory(self):
        np.testing.assert_array_equal(memorize(np.array([0,0,1]),np.array([0,1,1]),3),[0,1,0])
    def test_repeat(self):
        for a,b in zip(sample(20,7),sample(20,7)):np.testing.assert_array_equal(a,b)
    def test_independent_seed(self):self.assertFalse(np.array_equal(sample(100,7)[1],sample(100,8)[1]))
    def test_invalid_sample(self):
        with self.assertRaises(ValueError):sample(0,1)

if __name__=='__main__':unittest.main()
