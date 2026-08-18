"""CPU checks for native Wan SpheRoPE support."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comfy.ldm.flux.layers import EmbedND
from native_nodes import _SFCWanEmbedND


class NativeWanSFCTests(unittest.TestCase):
    def setUp(self):
        self.original = EmbedND(dim=128, theta=10000, axes_dim=[44, 42, 42])
        self.adapter = _SFCWanEmbedND(self.original)

    def _ids(self, frames=2, rows=4, columns=8):
        t, h, w = torch.meshgrid(
            torch.arange(frames, dtype=torch.float32),
            torch.arange(rows, dtype=torch.float32),
            torch.arange(columns, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack((t, h, w), dim=-1).reshape(1, -1, 3)

    def test_preserves_wan_shape_and_non_width_axes(self):
        ids = self._ids()
        stock = self.original(ids)
        adapted = self.adapter(ids)
        self.assertEqual(stock.shape, adapted.shape)
        prefix = (44 + 42) // 2
        self.assertTrue(torch.equal(stock[..., :prefix, :, :], adapted[..., :prefix, :, :]))
        self.assertFalse(torch.equal(stock[..., prefix:, :, :], adapted[..., prefix:, :, :]))

    def test_zero_strength_matches_stock_rope(self):
        ids = self._ids()
        stock = self.original(ids)
        adapted = _SFCWanEmbedND(self.original, strength=0.0)(ids)
        self.assertTrue(torch.allclose(stock, adapted, atol=1e-6))
    def test_spherical_path_converges_at_erp_poles(self):
        rows, columns = 5, 16
        full_sfc = _SFCWanEmbedND(self.original, strength=1.0)
        matrices = full_sfc._width_matrices(rows, columns, torch.device("cpu"), torch.float32)
        frequencies = 1.0 / (10000.0 ** torch.linspace(0, 40 / 42, steps=21, dtype=torch.float64))
        path_b = columns * frequencies / (2.0 * torch.pi) < 1.0
        self.assertTrue(path_b.any())
        south = matrices[0, :, path_b]
        north = matrices[-1, :, path_b]
        self.assertTrue(torch.allclose(south, south[:1].expand_as(south), atol=1e-6))
        self.assertTrue(torch.allclose(north, north[:1].expand_as(north), atol=1e-6))

    def test_only_integrated_nodes_are_public(self):
        from native_nodes import NODE_CLASS_MAPPINGS

        self.assertEqual(
            set(NODE_CLASS_MAPPINGS),
            {"SpheRoPEERPConditioning", "SpheRoPEPipelinePatch"},
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
