import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from twincore import Params, simulate, sample_params

class TwinWorldTests(unittest.TestCase):
    def test_observational_equivalence_all_random_seeds(self):
        for seed in range(30):
            p = sample_params(np.random.default_rng(seed))
            a, _ = simulate(p, coupled=False, force=False)
            b, _ = simulate(p, coupled=True, force=False)
            self.assertTrue(np.array_equal(a,b))

    def test_intervention_separates_mechanisms(self):
        for seed in range(30):
            p = sample_params(np.random.default_rng(seed))
            a, u = simulate(p, coupled=False, force=True)
            b, v = simulate(p, coupled=True, force=True)
            self.assertTrue(np.array_equal(u,v))
            self.assertGreater(float(np.max(np.abs(a-b))), 1e-6)
            self.assertGreater(float(np.max(np.abs(a[:,1]-b[:,1]))), 1e-6)

    def test_no_action_affects_independent_b(self):
        p = Params(1,1,8,.8,.1,1,.2,5)
        passive, _ = simulate(p,coupled=False,force=False)
        active, _ = simulate(p,coupled=False,force=True)
        self.assertTrue(np.array_equal(passive[:,1],active[:,1]))

if __name__ == '__main__':
    unittest.main()
