import unittest
import sys; sys.path.append('../')

from openquake.hazardlib.source import SimpleFaultSource

import hztest


class TestBasicUtils(unittest.TestCase):

    def test_flatten_list(self):
        lol = [['l'], ['o'], ['l']]
        flol = hztest.utils.utils._flatten_list(lol)
        self.assertEqual(flol, ['l', 'o', 'l'])

class TestPHL1(unittest.TestCase):

    def setUp(self):
        self.test_dir = './data/source_models/sm1/'
        self.lt = hztest.utils.io.process_logic_tree(self.test_dir)

    def test_process_logic_tree(self):
        test_lt = {'b1': {'area': [],
                          'complex_fault': [],
                          'point': [],
                          'multipoint': [],
                          'none': [None]
                          }}

        for k, v in test_lt['b1'].items():
            self.assertEqual(self.lt['b1'][k], v)

        self.assertIsInstance(self.lt['b1']['simple_fault'][0], 
                              SimpleFaultSource)


if __name__ == '__main__':
    unittest.main()