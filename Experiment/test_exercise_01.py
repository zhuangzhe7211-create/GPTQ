"""验收数学行为。TODO 未完成时测试失败是预期，不含参考实现。"""
import unittest
from exercise_01 import weighted_error, best_candidate


class WeightedObjectiveTests(unittest.TestCase):
    def test_exact_value(self):
        self.assertAlmostEqual(weighted_error([1, 3], [0, 1], [2, 4]), 9.0)

    def test_zero_error(self):
        self.assertEqual(weighted_error([1, -2], [1, -2], [9, 1]), 0)

    def test_constant_positive_scaling_preserves_choice(self):
        args = ([-2, -1, 0, 1, 2], [1, 1], [0, 2])
        self.assertEqual(best_candidate(*args, [1, 1]), 1)
        self.assertEqual(best_candidate(*args, [7, 7]), 1)

    def test_token_weighting_changes_choice(self):
        args = ([-2, -1, 0, 1, 2], [1, 1], [0, 2])
        self.assertEqual(best_candidate(*args, [9, 1]), 0)
        self.assertEqual(best_candidate(*args, [1, 9]), 2)

    def test_nontrivial_inputs_and_tie_order(self):
        self.assertEqual(best_candidate([-1, 0, 1, 2], [2, -1], [4, -2], [1, 1]), 2)
        self.assertEqual(best_candidate([2, 0], [1], [1], [1]), 2)


if __name__ == '__main__':
    unittest.main()
