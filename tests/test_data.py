"""
Tests for data.py
"""

import unittest
import data
from tests import core # shared test data

class TestDataset(unittest.TestCase):
    """
    Tests for Dataset class
    """

    def test_make(self):
        """
        Return a dataloader from the default dataset.
        """
        dataloader = data.make(core.cfg)
        self.assertIsNotNone(dataloader)
        dataiter = iter(dataloader)
        x, y = next(dataiter)
        self.assertEqual(x.shape, y.shape)
