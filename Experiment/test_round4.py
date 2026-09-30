import unittest
import torch
from build_round4_prefix import gptq_h
from quant_core import reference_nd, RowQuantizer


class PrefixChecks(unittest.TestCase):
    def test_block_kernel_matches_independent_column_reference(self):
        torch.manual_seed(52)
        x = torch.randn(19, 100)
        w = torch.randn(11, 19)
        expected = reference_nd(w, x, x, torch.ones(100), alpha=0., beta=0.)
        for blocksize in (1, 7, 128):
            result = gptq_h(w, x@x.T/100, blocksize=blocksize)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            torch.testing.assert_close(result, RowQuantizer(w,4).quantize(result))


if __name__ == '__main__':
    unittest.main()
