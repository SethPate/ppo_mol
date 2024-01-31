"""
Tests for the transformer model code.
"""

import unittest
import model
import torch
import core

class TestModels(unittest.TestCase):

    def setUp(self):
        self.test_cfg = core.cfg

    def test_make(self):
        cfg['model_f'] = 'example_path'
        tf = model.make(cfg)
        self.assertIsNotNone(tf)

    def test_init(self):
        """
        Create a transformer object.
        """
        tf = model.Transformer(self.test_cfg)
        self.assertTrue(True)

    def test_fwd(self):
        """
        Try to run something through the transformer.
        """
        tf = model.Transformer(self.test_cfg)
        tf.eval()
        # input of size (batch, seq_len, 1)
        x = torch.randint(40, (1,40))
        y = tf(x)
        self.assertEqual(y.shape, (1,40,40))

    def test_sample(self):
        """
        Sample randomly generated data.
        """
        tf = model.Transformer(self.test_cfg)
        tf.eval()
        y = tf.sample(2, max_new=10)
        self.assertEqual(len(y), 2)
