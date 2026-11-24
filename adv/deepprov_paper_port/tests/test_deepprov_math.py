import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deepprov_math import act_on_numpy, tradeoff_score, density_peak_reference, summarize_per_channel
from deepprov_io import epsilon_tag

class MathTest(unittest.TestCase):
    def test_activation_official_semantics(self):
        x=np.array([[2.,3.]],np.float32)
        r=np.array([1.,1.],np.float32)
        beta=np.array([2.,1.],np.float32)
        mask=np.array([True,True])
        a=act_on_numpy(x,r,beta,mask,alpha=.5)
        np.testing.assert_array_almost_equal(a,[[1.5,3.]])
    def test_nonselected_node_unchanged(self):
        x=np.array([[10.,20.]],np.float32)
        out=act_on_numpy(x,[0.,0.],[100.,100.],[1.,0.],alpha=1.)
        np.testing.assert_array_equal(out,[[0.,20.]])
    def test_beta_strict_less_than(self):
        x=np.array([[1.]],np.float32)
        out=act_on_numpy(x,[0.],[1.],[1.],alpha=1.)
        np.testing.assert_array_equal(out,x)
    def test_tradeoff(self):
        self.assertAlmostEqual(tradeoff_score(.95,.30,.93,.40),.08)
        self.assertLess(tradeoff_score(.95,.30,.80,.40),0)
    def test_kde(self):
        samples=np.concatenate([np.ones(100),np.ones(10)*5.])
        v=density_peak_reference(samples,seed=3)
        self.assertTrue(.8<v<2.0,v)
    def test_candidate_and_beta(self):
        rng=np.random.RandomState(42)
        clean=np.abs(rng.normal(size=(32,4))).astype(np.float32)
        adv=clean.copy();adv[:,1]+=3
        attrs=np.ones(4)
        candidates, ref, beta=summarize_per_channel(clean,adv,attrs,attrs,reference_mode='mean')
        self.assertEqual(ref.shape,(4,))
        self.assertEqual(beta.shape,(4,))
        self.assertTrue(any(c['neuron_idx']==1 for c in candidates))
    def test_epsilon_file(self):
        self.assertEqual(epsilon_tag(.05),'epsilon0.05')
        self.assertEqual(epsilon_tag(.1),'epsilon0.1')

if __name__=='__main__': unittest.main()
