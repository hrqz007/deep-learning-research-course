import unittest
import numpy as np
import experiment as e
class TestAudio(unittest.TestCase):
    def test_shape(self): self.assertEqual(e.stft(np.zeros((3,4096))).shape,(3,61,129))
    def test_short_input(self):
        with self.assertRaises(ValueError): e.stft(np.zeros((1,255)))
    def test_pair_balanced(self):
        x,y,g,p=e.generate(sources=4,pairs=2)
        for source in np.unique(g): self.assertEqual(int(y[g==source].sum()),4//2)
        self.assertEqual(len(set(g[g<2])&set(g[g>=2])),0)
    def test_chunk_swap(self):
        x,*_=e.generate(sources=1,pairs=1)
        np.testing.assert_array_equal(x[0,:2048],x[1,2048:])
        np.testing.assert_array_equal(x[0,2048:],x[1,:2048])
    def test_pooled_invariant(self):
        b=np.random.default_rng(1).normal(size=(8,61,2))
        np.testing.assert_allclose(e.features(b,False),e.features(e.shuffle_frames(b,3),False),atol=1e-14)
    def test_alias(self):
        t=np.arange(100)/8000
        np.testing.assert_allclose(np.sin(2*np.pi*1000*t),-np.sin(2*np.pi*7000*t),atol=1e-12)
    def test_alignment(self):
        start,center,f=e.alignment_report()
        self.assertEqual(start[-1],3840);self.assertEqual(len(center),61)
        self.assertAlmostEqual(f[30],.5);self.assertAlmostEqual(f[31],.75)
    def test_features(self): self.assertEqual(e.features(np.ones((2,61,2))).shape,(2,8))
    def test_learns(self):
        x,y,g,_=e.generate(sources=6,pairs=2);b=e.band_features(e.stft(x));f=e.features(b)
        model=e.fit_logistic(f[g<4],y[g<4],steps=100)
        self.assertGreater(np.mean((e.predict(model,f[g>=4])>=.5)==y[g>=4]),.9)
    def test_normalization_training_only(self):
        x=np.array([[1.,2.],[3.,4.]]);m=e.fit_logistic(x,np.array([0,1]),steps=2)
        np.testing.assert_array_equal(m['mu'],[2,3])
if __name__=='__main__':unittest.main()
