"""Independent exact and 90-digit numerical oracles for unit 017.

The oracle does not call production density/likelihood inside its reference path.
Run with `python -m unittest -v test_experiment.py` from this unit directory.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from pathlib import Path
from unittest.mock import patch
import json
import math
import subprocess
import sys
import tempfile
import unittest
import experiment as ex


def decimal_pi():
    """Machin formula from independent arctan power series, 90-digit context."""
    with localcontext() as ctx:
        ctx.prec = 90
        def arctan_inverse(n):
            x = Decimal(1) / n
            power, total, k, sign = x, Decimal(0), 0, 1
            while True:
                term = power / (2 * k + 1)
                total += term if sign == 1 else -term
                if abs(term) < Decimal('1e-88'):
                    return total
                power *= x * x
                sign = -sign
                k += 1
        return +(16 * arctan_inverse(5) - 4 * arctan_inverse(239))


PI = decimal_pi()


def gaussian_oracle(x, mu=0, sigma=1):
    with localcontext() as ctx:
        ctx.prec = 90
        x, mu, sigma = [Decimal.from_float(float(v)) for v in (x, mu, sigma)]
        z = (x - mu) / sigma
        density = (-z * z / 2).exp() / (sigma * (2 * PI).sqrt())
        return +density, +density.ln()


def symmetric_mass_oracle(a):
    """Integrate exp(-z²/2) on [-a,a] by its power series, not quadrature.

    For a<=6 the tail is alternating and decreasing long before termination.
    The next-term bound is below 1e-85 in the 90-digit context.
    """
    with localcontext() as ctx:
        ctx.prec = 90
        a = Decimal(a)
        term = a
        total = term
        for k in range(5000):
            term *= -(a * a) * (2 * k + 1) / (2 * (k + 1) * (2 * k + 3))
            total += term
            if abs(term) < Decimal('1e-85'):
                return +(2 * total / (2 * PI).sqrt())
        raise AssertionError('oracle series did not converge')


class ProbabilityTests(unittest.TestCase):
    def test_01_finite_distribution_and_exact_likelihood(self):
        count = 0
        for denominator in (2, 4, 8):
            for numerator in range(denominator + 1):
                p = Fraction(numerator, denominator)
                for n in range(1, 7):
                    exact_total = Fraction(0)
                    for sequence in product((0, 1), repeat=n):
                        reference = Fraction(1)
                        for y in sequence:
                            reference *= p if y else 1 - p
                        exact_total += reference
                        got = ex.bernoulli_likelihood(sequence, float(p))
                        self.assertEqual(got, float(reference))  # dyadic is exact here
                        count += 1
                    self.assertEqual(exact_total, 1)
        self.assertEqual(count, 2142)
        self.assertEqual(ex.bernoulli_likelihood([1, 1, 1, 0], .25), 3 / 256)
        self.assertEqual(ex.bernoulli_likelihood([1, 1, 1, 0], .75), 27 / 256)
        count_event = sum(ex.bernoulli_likelihood(s, .75) for s in product((0, 1), repeat=4) if sum(s) == 3)
        self.assertEqual(count_event, 27 / 64)

    def test_02_log_likelihood_decimal_and_endpoints(self):
        with localcontext() as ctx:
            ctx.prec = 90
            for p in (.125, .25, .5, .75, .875, 1e-20, math.nextafter(1.0, 0.0)):
                dp = Decimal.from_float(p)
                for values in ([0, 0], [1, 1, 1], [1, 1, 1, 0]):
                    ref = sum((dp if y else 1 - dp).ln() for y in values)
                    self.assertAlmostEqual(ex.bernoulli_log_likelihood(values, p), float(ref), delta=2e-13)
        for values, p, expected in [([0, 0], 0, 0), ([1, 1], 1, 0), ([1, 0], 0, -math.inf), ([1, 0], 1, -math.inf)]:
            self.assertEqual(ex.bernoulli_log_likelihood(values, p), expected)
        self.assertEqual(ex.bernoulli_likelihood([1] * 2000, .5), 0)
        self.assertAlmostEqual(ex.bernoulli_log_likelihood([1] * 2000, .5), -2000 * math.log(2), places=11)

    def test_03_finite_generator_exhaustive_slots_and_seed(self):
        class AllSlots:
            def __init__(self, seed):
                self.index = 0
            def randrange(self, slots):
                value = self.index % slots
                self.index += 1
                return value
        with patch.object(ex.random, 'Random', AllSlots):
            for slots in range(1, 21):
                for marked in range(slots + 1):
                    generated = ex.bernoulli_sample(slots, marked, slots, 17)
                    self.assertEqual(sum(generated), marked)
                    self.assertEqual(generated, [1] * marked + [0] * (slots - marked))
        self.assertEqual(ex.bernoulli_sample(12, 3, 4, 17), [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1])
        self.assertEqual(ex.bernoulli_sample(100, 3, 4, 17), ex.bernoulli_sample(100, 3, 4, 17))
        self.assertEqual(ex.bernoulli_sample(4, 0, 7), [0] * 4)
        self.assertEqual(ex.bernoulli_sample(4, 7, 7), [1] * 4)
        self.assertEqual(ex.bernoulli_sample(0, 3, 4), [])

    def test_04_finite_bayes_independent_fraction(self):
        values = [1, 1, 1, 0]
        candidates = [Fraction(1, 4), Fraction(3, 4)]
        for weights in ([Fraction(1, 2)] * 2, [Fraction(9, 10), Fraction(1, 10)]):
            scores = [w * p**3 * (1-p) for p, w in zip(candidates, weights)]
            reference = [v / sum(scores) for v in scores]
            got = ex.finite_posterior(values, list(map(float, candidates)), list(map(float, weights)))
            for a, b in zip(got, reference):
                self.assertAlmostEqual(a, float(b), places=14)
        self.assertEqual(ex.finite_posterior([1], [0, 1], [.5, .5]), [0, 1])
        # Scores underflow in product space, but the posterior is still defined.
        self.assertEqual(ex.finite_posterior([1] * 2000, [.25, .5], [.5, .5]), [0, 1])

    def test_05_gaussian_point_decimal_oracle(self):
        self.assertAlmostEqual(float(PI), math.pi, places=15)
        fixtures = 0
        for mu in (-1.25, 0., 2.):
            for sigma in (.1, .5, 1., 4.):
                for z in (-5., -2., -.25, 0., .75, 3.):
                    x = mu + sigma * z
                    value, log_value = gaussian_oracle(x, mu, sigma)
                    self.assertAlmostEqual(ex.gaussian_density(x, mu, sigma), float(value), delta=4e-14 * max(1, float(value)))
                    self.assertAlmostEqual(ex.gaussian_log_density(x, mu, sigma), float(log_value), delta=3e-14)
                    fixtures += 1
        self.assertEqual(fixtures, 72)
        self.assertGreater(ex.gaussian_density(0, 0, .1), 1)
        self.assertEqual(ex.gaussian_density(40), 0)
        self.assertAlmostEqual(ex.gaussian_log_density(40), float(gaussian_oracle(40)[1]), places=12)
        self.assertTrue(math.isfinite(ex.gaussian_log_density(0, 0, 1e308)))
        self.assertGreater(ex.gaussian_density(0, 0, 1e308), 0)

    def test_06_integrals_exact_and_high_precision(self):
        for panels in (1, 2, 4, 16, 100):
            # Exact independent integral of 3x+2 over [-1,2] is 21/2.
            reference = Fraction(3, 2) * (Fraction(2)**2 - Fraction(-1)**2) + 2 * (2 - (-1))
            self.assertAlmostEqual(ex.midpoint_integral(lambda x: 3*x+2, -1, 2, panels), float(reference), places=12)
        self.assertEqual(ex.midpoint_integral(lambda x: 4, 0, .25, 100), 1)
        for bound in (1, 2, 4, 6):
            expected = float(symmetric_mass_oracle(bound))
            estimate = ex.midpoint_integral(ex.gaussian_density, -bound, bound, 2000)
            self.assertAlmostEqual(estimate, expected, delta=1.5e-7)
        self.assertAlmostEqual(float(symmetric_mass_oracle(1)), .6826894921370859, places=15)
        self.assertAlmostEqual(float(symmetric_mass_oracle(2)), .9544997361036416, places=15)
        self.assertLess(ex.midpoint_integral(lambda x: ex.gaussian_density(x,0,.01), -1, 1, 20), .00003)
        self.assertAlmostEqual(ex.midpoint_integral(lambda x: ex.gaussian_density(x,0,.01), -1, 1, 2000), 1, places=13)

    def test_07_gaussian_likelihood_units_and_degeneracy(self):
        values = [-.1, .1]
        a = ex.gaussian_log_likelihood(values, 0, .1)
        b = ex.gaussian_log_likelihood(values, .1, .1)
        self.assertAlmostEqual(a-b, 1, places=14)
        self.assertAlmostEqual(ex.gaussian_log_likelihood([-100,100], 0, 100), a-2*math.log(1000), places=13)
        self.assertAlmostEqual(ex.gaussian_density(100,20,50), ex.gaussian_density(.1,.02,.05)/1000, places=14)
        self.assertGreater(ex.gaussian_density(1,1,.01), ex.gaussian_density(1,1,.1))
        self.assertAlmostEqual(ex.gaussian_log_likelihood([1,1],1,.5), 2*ex.gaussian_log_density(1,1,.5), places=14)

    def test_08_invalid_inputs_and_numeric_failure(self):
        rejected = []
        calls = []
        for p in (-.01, 1.01, math.nan, math.inf, -math.inf, True, '0.5', [0.5], 10**1000):
            calls.append(lambda p=p: ex.bernoulli_log_likelihood([1,0], p))
        for values in ([], [True], [0.,1], [[1]], [2], '10', [math.nan], [1,None]):
            calls.append(lambda values=values: ex.bernoulli_log_likelihood(values, .5))
        for args in ((-1,1,2), (1,-1,2), (1,3,2), (1,1,0), (True,1,2), (1,True,2), (1,1,2.)):
            calls.append(lambda args=args: ex.bernoulli_sample(*args))
        calls.append(lambda: ex.bernoulli_sample(1,1,2, True))
        for args in ((0,0,0), (0,0,-1), (True,0,1), (0,math.inf,1), (0,0,math.nan), (1e308,-1e308,1), (1,0,1e-308), (1e200,0,1)):
            calls.append(lambda args=args: ex.gaussian_log_density(*args))
        calls.extend([
            lambda: ex.gaussian_density(0,0,5e-324),
            lambda: ex.gaussian_log_likelihood([],0,1),
            lambda: ex.gaussian_log_likelihood([[1]],0,1),
            lambda: ex.midpoint_integral(lambda x: 1.,0,1,0),
            lambda: ex.midpoint_integral(lambda x: 1.,0,1,True),
            lambda: ex.midpoint_integral(lambda x: 1.,1,0,10),
            lambda: ex.midpoint_integral(lambda x: 1.,-1e308,1e308,10),
            lambda: ex.midpoint_integral(lambda x: 1.,1e16,1e16+4,100),
            lambda: ex.midpoint_integral(lambda x: math.nan,0,1,10),
            lambda: ex.midpoint_integral(lambda x: 1e308,0,100,10),
            lambda: ex.finite_posterior([1],[0,0],[.5,.5]),
            lambda: ex.finite_posterior([1],[0,1],[.2,.3]),
            lambda: ex.finite_posterior([1],[0,1],[-.1,1.1]),
            lambda: ex.finite_posterior([1,0],[0,1],[.5,.5]),
        ])
        for call in calls:
            with self.assertRaises((ValueError, TypeError, ArithmeticError)):
                call()
            rejected.append(1)
        self.assertEqual(len(rejected), 47)
        seen=[]
        with self.assertRaises(ArithmeticError):
            ex.midpoint_integral(lambda x: seen.append(x) or 1, -1e308, 1e308, 5)
        self.assertEqual(seen, [])

    def test_09_csv_contract_and_fresh_process_reproducibility(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for bad in ('trial,y\n1,True\n', 'trial,y\n2,1\n', 'trial,y\n', 'trial,y\n1,1.0\n',
                        'trial,y,y\n1,0,1\n2,0,1\n3,0,1\n4,1,0\n',
                        'trial,trial,y\n9,1,1\n',
                        'trial,y,extra\n1,1,x\n'):
                path = temp/'bad.csv'
                path.write_text(bad)
                with self.assertRaises(ValueError): ex.load_binary_csv(path)
            # Duplicate headers must fail before any output creation or old-file overwrite.
            import shutil
            clone = temp / 'clone'
            (clone / 'data').mkdir(parents=True)
            shutil.copyfile(root / 'experiment.py', clone / 'experiment.py')
            old = temp / 'old'
            old.mkdir()
            names = ('bernoulli.csv','gaussian_integrals.csv','summary.json')
            for name in names: (old/name).write_bytes(b'unchanged sentinel')
            for malformed in ('trial,y,y\n1,0,1\n2,0,1\n3,0,1\n4,1,0\n',
                              'trial,trial,y\n9,1,1\n', 'trial,y,extra\n1,1,x\n'):
                (clone/'data/observations.csv').write_text(malformed)
                for target in (old, temp/'new'):
                    result = subprocess.run([sys.executable, str(clone/'experiment.py'), '--output-dir', str(target)], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('expected exact unique CSV header', result.stderr)
                self.assertFalse((temp/'new').exists())
                for name in names: self.assertEqual((old/name).read_bytes(), b'unchanged sentinel')
            for folder in ('a','b'):
                result = subprocess.run([sys.executable, str(root/'experiment.py'), '--output-dir', str(temp/folder)], capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(result.stdout)['successes'], 3)
            for name in ('bernoulli.csv','gaussian_integrals.csv','summary.json'):
                self.assertEqual((temp/'a'/name).read_bytes(), (temp/'b'/name).read_bytes())
                self.assertEqual((temp/'a'/name).read_bytes(), (root/'outputs'/name).read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
