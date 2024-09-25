# -*- coding: utf-8 -*-

import unittest

class DuplyTestSuite(unittest.TestCase):
    """Basic test cases."""


    def test_relative_truth_and_lies(self):
        assert True


    def test_relative_truth_and_meaning(self):
        assert True == False, "Tests are working when you get this error"


if __name__ == '__main__':
    unittest.main()