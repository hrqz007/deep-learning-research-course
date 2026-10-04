"""Independent expanded-polynomial Fraction and Decimal oracles, plus failures."""
from fractions import Fraction as F
from decimal import Decimal as D, localcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import re
import numpy as np
import experiment as lab

# L = x^2*y^2 + 2*x^2*y + x^2; no graph or lab.gradient reuse.
TERMS = [(F(1), 2, 2), (F(2), 2, 1), (F(1), 2, 0)]


def oracle(x, y):
    val = sum(c * x ** i * y ** j for c, i, j in TERMS)
    gx = sum(c * i * x ** (i - 1) * y ** j for c, i, j in TERMS if i)
    gy = sum(c * j * x ** i * y ** (j - 1) for c, i, j in TERMS if j)
    return val, gx, gy


def expect_failure(call, allowed=(ValueError, TypeError, ArithmeticError)):
    try:
        call()
    except allowed:
        return
    raise AssertionError("Invalid input was accepted")


def run_tests():
    groups = []
    p = [[2, 3]]
    assert lab.forward(p) == {"x": 2, "y": 3, "a": 6, "z": 8, "loss": 64}
    assert lab.gradient(p).shape == (1, 2)
    np.testing.assert_array_equal(lab.gradient(p), [[64, 32]])
    assert lab.path_contributions(p) == {"x_via_a": 48, "x_direct": 16, "x_total": 64, "y_total": 32}
    groups.append("fixed hand calculations and shared-variable path accumulation")
    count = 0
    for numerator_x in range(-4, 5):
        for numerator_y in range(-4, 5):
            x, y = F(numerator_x, 2), F(numerator_y, 4)
            val, gx, gy = oracle(x, y)
            point = [[float(x), float(y)]]
            assert lab.loss(point) == float(val)
            np.testing.assert_allclose(lab.gradient(point), [[float(gx), float(gy)]], rtol=0, atol=1e-13)
            h = F(1, 8)
            dx = (oracle(x + h, y)[0] - oracle(x - h, y)[0]) / (2 * h)
            dy = (oracle(x, y + h)[0] - oracle(x, y - h)[0]) / (2 * h)
            assert (dx, dy) == (gx, gy)
            np.testing.assert_allclose(lab.central_gradient(lab.loss, point, float(h)), [[float(dx), float(dy)]], rtol=0, atol=1e-13)
            count += 1
    groups.append("81 independent rational expanded-polynomial loss, gradient and central differences")
    for h in [F(1, 2), F(1, 10), F(1, 100), F(1, 1000)]:
        exact = (oracle(2 + h, 3 + h)[0] - oracle(2 - h, 3 - h)[0]) / (2 * h)
        assert exact == 96 + 12 * h * h
        assert abs(lab.central_curve(2, float(h)) - float(exact)) < 2e-10
    with localcontext() as ctx:
        ctx.prec = 80
        h = D('1e-20')
        f = lambda t: t**4 + 4*t**3 + 4*t**2
        numeric = (f(D(2) + h) - f(D(2) - h)) / (2*h)
        assert abs(numeric - (D(96) + 12*h*h)) < D('1e-55')
        sqrt5 = D(5).sqrt()
        assert abs(D.from_float(math.sqrt(5120)) - 32*sqrt5) < D('2e-14')
    groups.append("curve central-error identity and 80-digit Decimal oracle")
    assert lab.directional_derivative(p, [[.6, .8]]) == 64
    assert lab.rate_along(p, [[3, 4]]) == 320
    assert lab.rate_along(p, [[1, 1]]) == 96
    assert abs(lab.directional_derivative(p, [[-1/math.sqrt(5), 2/math.sqrt(5)]])) < 1e-13
    groups.append("unit direction versus velocity and orthogonal tangent")
    for e in [F(1, 10), F(1, 100), F(1, 1000)]:
        result = lab.linear_prediction(p, [[float(e), float(2*e)]])
        exact = oracle(2+e, 3+2*e)[0]
        remainder = 96*e*e + 32*e**3 + 4*e**4
        assert exact - 64 - 128*e == remainder
        assert abs(result["actual_loss"] - float(exact)) < 1e-12
        assert abs(result["remainder"] - float(remainder)) < 1e-12
    for eta in [F(1, 1000), F(1, 100), F(1, 10), F(1, 2)]:
        actual = lab.gradient_step(p, float(eta))
        x, y = 2 - 64*eta, 3 - 32*eta
        assert abs(actual["new_loss"] - float(oracle(x, y)[0])) < 1e-9
    assert lab.gradient_step(p, .5)["new_loss"] == 129600
    groups.append("exact local remainder and finite negative-gradient overshoot")
    zero = [[0, 0]]
    np.testing.assert_array_equal(lab.central_gradient(lab.partials_not_enough, zero, .01), zero)
    for e in [.1, .01, .001]:
        assert abs(lab.partials_not_enough([[e, e]])/e - 1/math.sqrt(2)) < 1e-15
        assert abs(lab.partials_not_enough([[-e, -e]])/(-e) + 1/math.sqrt(2)) < 1e-15
    groups.append("partial existence cannot establish differentiability")
    failures = []
    for bad in [[2, 3], [[2], [3]], [[[2, 3]]], [], [[True, False]], [[True, 3]], [[2, np.bool_(False)]], [[np.array(True), 3.]], [[2., np.array(False)]], [[1+0j, 2]], [["2", "3"]], [[float('nan'), 3]], [[2, float('inf')]]]:
        failures.append(lambda bad=bad: lab.gradient(bad))
    for h in [0, -1, float('nan'), float('inf'), True, [0.1], 1e-18]:
        failures.append(lambda h=h: lab.central_gradient(lab.loss, p, h))
    failures += [lambda: lab.directional_derivative(p, [[3, 4]]),
                 lambda: lab.directional_derivative(p, [[0, 0]]),
                 lambda: lab.gradient_step(p, 0),
                 lambda: lab.gradient_step(p, -1),
                 lambda: lab.loss([[1e308, 1e308]]),
                 lambda: lab.central_gradient(lab.loss, [[1e308, 2]], 1e308),
                 lambda: lab.central_gradient(lambda q: [1, 2], p, .1),
                 lambda: lab.central_gradient(lambda q: float('nan'), p, .1),
                 lambda: lab.central_curve(2, 1e-18),
                 lambda: lab.central_curve(2, -1),
                 lambda: lab.rate_along(p, [[1e308, 1e308]]),
                 lambda: lab.partials_not_enough([[1.7e308, 1.7e308]])]
    for case in failures: expect_failure(case)
    original = np.array([[2., 3.]])
    saved = original.copy()
    lab.central_gradient(lab.loss, original, .01)
    np.testing.assert_array_equal(original, saved)
    groups.append(f"{len(failures)} shape, type, finite-value, step, norm and overflow rejections; no input mutation")
    with TemporaryDirectory() as tmp:
        a, b = Path(tmp)/"a", Path(tmp)/"b"
        assert lab.run(a) == lab.run(b)
        for path in a.iterdir(): assert path.read_bytes() == (b/path.name).read_bytes()
    with TemporaryDirectory() as tmp:
        old_base = lab.BASE
        config = json.loads((old_base/"data/experiment_config.json").read_text())
        config["point"] = [[1, 1]]
        root = Path(tmp)
        (root/"data").mkdir()
        (root/"data/experiment_config.json").write_text(json.dumps(config))
        try:
            lab.BASE = root
            expect_failure(lambda: lab.run(root/"outputs"), (ValueError,))
            assert not (root/"outputs").exists()
        finally:
            lab.BASE = old_base
    groups.append("full repeated run is byte-identical; altered fixed-report point rejected before writing")
    snippets = 0
    for name in ["lecture.md", "lab.md", "answers.md", "README.md"]:
        path = lab.BASE/name
        if not path.exists(): continue
        blocks = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
        namespace = {"np": np, "lab": lab}
        for block in blocks:
            exec(compile(block, str(path), "exec"), namespace)
            snippets += 1
    groups.append(f"all {snippets} Python document snippets execute in order")
    return {"status": "passed", "groups": len(groups), "test_groups": groups,
            "rational_fixtures": count, "rejected_cases": len(failures), "python_snippets": snippets}


if __name__ == "__main__":
    print(json.dumps(run_tests(), ensure_ascii=False, indent=2))
