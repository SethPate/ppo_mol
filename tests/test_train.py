"""
Tests for the training script: pretraining, finetuning, and evaluation.
"""

import unittest
import train
from tests import core

class TestTrain(unittest.TestCase):

    def test_pretrain(self):
        """
        Run the pretraining script.
        """
        train.main(core.cfg)
