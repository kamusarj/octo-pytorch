import unittest

import numpy as np
import torch

from octo.utils.train_utils_pt import _np2pt


class NumpyToTorchTest(unittest.TestCase):
    def test_preserves_string_metadata_and_converts_numeric_statistics(self):
        converted = _np2pt(
            {
                "dataset_fingerprint": np.array("abc123"),
                "action": {"mean": np.array([1.0, 2.0], dtype=np.float32)},
            }
        )

        self.assertEqual(converted["dataset_fingerprint"].item(), "abc123")
        self.assertTrue(torch.equal(converted["action"]["mean"], torch.tensor([1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
