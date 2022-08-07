import unittest

import numpy as np

from utils import valve_limit


class UtilTestCase(unittest.TestCase):
    mask_test = np.zeros((1024, 1024), dtype=np.uint8)
    mask_test[0:20, :] = 1

    def test_valve_limit(self):
        mask_result = valve_limit(self.mask_test, max_valve_num=49)
        self.assertTrue(np.all(np.sum(mask_result, 1) <= 49))  # add assertion here


if __name__ == '__main__':
    unittest.main()
