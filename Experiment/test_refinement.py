"""Check that tuning uses one shared alpha rather than the best per seed."""
import unittest

from evaluate_refinement import select_methods


class SelectionTest(unittest.TestCase):
    def test_select_by_mean_validation_ce(self):
        runs = []
        for losses in ((1., 2., 3.), (5., 2., 3.), (1., 2., 3.)):
            runs.append({f'{family}_a{alpha:g}': {'ce': loss}
                         for family in ('rescomp', 'gptaq', 'teacher', 'prefix', 'shared')
                         for alpha, loss in zip((.25, .5, 1.), losses)})
        selected, scores = select_methods(runs)
        self.assertTrue(all(key.endswith('_a0.5') for key in selected.values()))
        self.assertEqual(scores['rescomp']['rescomp_a0.5'], 2.)


if __name__ == '__main__':
    unittest.main()
