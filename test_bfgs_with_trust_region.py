import unittest

import numpy as np

from bfgs_with_trust_region import BFGS_With_Trust_Region


class TestClass(unittest.TestCase):
    def setUp(self):
        print()
        print('*'*5 + 'Test Begins' + '*'*5)

    def test_case1(self):
        obj = BFGS_With_Trust_Region(init_point=[[0.3], [0.4]], func_idx=1)
        obj.solve()
        self.assertEqual(obj.is_converged(), True)

    def test_case2(self):
        obj = BFGS_With_Trust_Region(func_idx=2)
        obj.solve()
        self.assertEqual(obj.is_converged(), True)

    def tearDown(self):
        print('*'*5 + 'Test Over' + '*'*5)
        print()


if __name__ == '__main__':
    unittest.main()
