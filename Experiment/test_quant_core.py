import unittest
import torch
from quant_core import make_weights, statistics, official_rescomp, grouped_official, reference_nd


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)
        self.x = torch.randn(6, 40)
        self.xf = self.x + .1 * torch.randn_like(self.x)
        self.w = torch.randn(8, 6)
        self.g = torch.randn(8, 40)
        self.s = torch.rand(40) + .1

    def test_weighted_stats_and_residual(self):
        h, c = statistics(self.xf, self.x, self.s)
        t = (self.xf * self.s) @ self.x.T / 40
        torch.testing.assert_close(t, h + c)
        current = self.w + .1 * torch.randn_like(self.w)
        r = self.w @ self.xf - current @ self.x
        torch.testing.assert_close((r * self.s) @ self.x.T / 40,
                                   current @ c + (self.w - current) @ t, atol=1e-6, rtol=1e-5)

    def test_cholesky_fusion_against_direct_suffix_solve(self):
        h, c = statistics(self.xf, self.x, self.s)
        h += .01 * h.diag().mean() * torch.eye(6)
        l = torch.linalg.cholesky(torch.linalg.inv(h))
        fused = torch.triu(c @ l, 1) @ l.T
        for k in range(5):
            expected = torch.linalg.solve(h[k+1:, k+1:], c[k, k+1:])
            torch.testing.assert_close(fused[k, k+1:], expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(torch.tril(fused), torch.zeros_like(fused))

    def test_all_one_group_equivalence(self):
        a = official_rescomp(self.w, self.xf, self.x, torch.ones(40), blocksize=2)
        b = grouped_official(self.w, self.xf, self.x, self.g, 4, 0., 4, .01, .25, 2)
        torch.testing.assert_close(a, b)

    def test_official_weighting_equals_input_whitening(self):
        a = official_rescomp(self.w, self.xf, self.x, self.s, blocksize=2)
        b = official_rescomp(self.w, self.xf*self.s.sqrt(), self.x*self.s.sqrt(), torch.ones(40), blocksize=2)
        torch.testing.assert_close(a, b)

    def test_reference_equals_single_block_official(self):
        a = official_rescomp(self.w, self.xf, self.x, self.s, blocksize=6)
        b = reference_nd(self.w, self.xf, self.x, self.s)
        torch.testing.assert_close(a, b)

    def test_constant_row_weight_null_and_zero_gradient_fallback(self):
        a = official_rescomp(self.w, self.xf, self.x, torch.ones(40), blocksize=2)
        b = official_rescomp(self.w, self.xf, self.x, torch.full((40,), 7.), blocksize=2)
        torch.testing.assert_close(a, b)
        for _, score in make_weights(torch.zeros_like(self.g), groups=4, rho=1.):
            torch.testing.assert_close(score, torch.ones_like(score))


if __name__ == '__main__':
    unittest.main(verbosity=2)
