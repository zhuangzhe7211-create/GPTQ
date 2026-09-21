"""Independent small algebra checks; no model, downloads or third-party libraries.

Run: python verify_weighted_math.py
This is not a ResComp implementation or an empirical quality benchmark.
"""
import math
import unittest


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mv(a, x):
    return [dot(row, x) for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def mm(a, b):
    return [[dot(row, col) for col in transpose(b)] for row in a]


def weighted_cross(a, b, s):
    return mm([[v * st for v, st in zip(row, s)] for row in a], transpose(b))


def solve2(h, b):
    det = h[0][0] * h[1][1] - h[0][1] * h[1][0]
    return [(h[1][1] * b[0] - h[0][1] * b[1]) / det,
            (h[0][0] * b[1] - h[1][0] * b[0]) / det]


def constrained_step(h, b, a):
    u = solve2(h, b)
    v = solve2(h, [1., 0.])
    return [ui + vi * (a - u[0]) / v[0] for ui, vi in zip(u, v)]


class WeightedMathTests(unittest.TestCase):
    def setUp(self):
        self.x = [[1., 2., -1.], [0., 1., 2.]]
        self.xf = [[1.2, 1.8, -.7], [.1, 1.3, 1.7]]
        self.s = [.5, 2., 1.5]

    def close(self, a, b, places=9):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=places)

    def test_whitening(self):
        h = weighted_cross(self.x, self.x, self.s)
        xp = [[v * math.sqrt(s) for v, s in zip(row, self.s)] for row in self.x]
        hp = mm(xp, transpose(xp))
        for row, rp in zip(h, hp):
            self.close(row, rp)

    def test_fixed_target_residual_identity(self):
        w0, w = [.7, -.4], [.6, -.2]
        residual = [dot(w0, xf) - dot(w, x)
                    for xf, x in zip(transpose(self.xf), transpose(self.x))]
        dx = [[a - b for a, b in zip(rf, rq)] for rf, rq in zip(self.xf, self.x)]
        h = weighted_cross(self.x, self.x, self.s)
        c = weighted_cross(dx, self.x, self.s)
        t = weighted_cross(self.xf, self.x, self.s)
        for hr, cr, tr in zip(h, c, t):
            self.close([a + b for a, b in zip(hr, cr)], tr)
        lhs = mv(self.x, [s * r for s, r in zip(self.s, residual)])
        rhs = [dot(w, cc) + dot([a - b for a, b in zip(w0, w)], tc)
               for cc, tc in zip(transpose(c), transpose(t))]
        self.close(lhs, rhs)

    def test_gradient_and_hessian_finite_difference(self):
        x, s, y, q = self.x, self.s, [.3, -.2, .8], [.2, -.1]
        h = weighted_cross(x, x, s)
        b = mv(x, [st * yt for st, yt in zip(s, y)])

        def objective(v):
            return .5 * sum(st * (dot(col, v) - yt) ** 2
                            for col, st, yt in zip(transpose(x), s, y))

        def gradient(v):
            return [a - bb for a, bb in zip(mv(h, v), b)]

        eps = 1e-5
        for j in range(2):
            qp, qm = q[:], q[:]
            qp[j] += eps
            qm[j] -= eps
            self.assertAlmostEqual((objective(qp) - objective(qm)) / (2 * eps),
                                   gradient(q)[j], places=8)
            for k in range(2):
                self.assertAlmostEqual((gradient(qp)[k] - gradient(qm)[k]) / (2 * eps),
                                       h[k][j], places=8)

    def test_constrained_solution_kkt_and_minimum(self):
        h = weighted_cross(self.x, self.x, self.s)
        for j in range(2):
            h[j][j] += .1
        b, a = [.8, -.3], .25
        d = constrained_step(h, b, a)
        self.assertAlmostEqual(d[0], a)
        self.assertAlmostEqual(mv(h, d)[1] - b[1], 0.)

        def f(v):
            return .5 * dot(v, mv(h, v)) - dot(b, v)

        for offset in [-1., -.1, .1, 1.]:
            candidate = [a, d[1] + offset]
            self.assertGreater(f(candidate), f(d))
            self.assertAlmostEqual(f(candidate) - f(d), .5 * h[1][1] * offset ** 2)

    def test_common_scaling_invariance(self):
        h = weighted_cross(self.x, self.x, self.s)
        h[0][0] += .1
        h[1][1] += .1
        b, c = [.8, -.3], 7.
        d = constrained_step(h, b, .25)
        scaled = constrained_step([[c * v for v in row] for row in h],
                                  [c * v for v in b], .25)
        self.close(d, scaled)

    def test_all_one_weights(self):
        h = weighted_cross(self.x, self.x, [1., 1., 1.])
        for row, expected in zip(h, mm(self.x, transpose(self.x))):
            self.close(row, expected)

    def test_known_curvature_toy_changes_candidate(self):
        # x_quant=[1,1], x_fp=[0,2], w0=1. Known downstream curvature,
        # NOT teacher gradient-square weights: teacher task gradient is zero.
        grid = [-1, 0, 1, 2]

        def plain(q):
            return .5 * (q ** 2 + (q - 2) ** 2)

        def task(q):
            return .5 * (9 * q ** 2 + (q - 2) ** 2)

        q_plain = min(grid, key=plain)
        q_task = min(grid, key=task)
        self.assertEqual((q_plain, q_task), (1, 0))
        self.assertEqual((task(q_plain), task(q_task)), (5., 2.))


if __name__ == '__main__':
    unittest.main(verbosity=2)
