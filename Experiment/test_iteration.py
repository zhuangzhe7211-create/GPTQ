import unittest

import torch
from quant_core import make_weights, grouped_official, official_rescomp


class IterationChecks(unittest.TestCase):
    def test_rms_scores_scaling_and_null_regression(self):
        g = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
        _, score = make_weights(g, groups=1, rho=.25, power=.5)[0]
        torch.testing.assert_close(score, torch.tensor([.875, 1., 1.125]))
        torch.testing.assert_close(score, make_weights(g*7, groups=1, rho=.25, power=.5)[0][1])
        torch.manual_seed(7)
        x = torch.randn(4, 16)
        xf = x + .1*torch.randn_like(x)
        w = torch.randn(4, 4)
        grad = torch.randn(4, 16)
        base = official_rescomp(w, xf, x, torch.ones(16), blocksize=2)
        null = grouped_official(w, xf, x, grad, 2, 0., 4, .01, .25, 2, power=.5)
        torch.testing.assert_close(base, null)


if __name__ == '__main__':
    unittest.main()
