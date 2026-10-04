"""Unit 017: finite Bernoulli likelihood and continuous Gaussian density.

Core uses only Python's standard library. All examples are synthetic.
No probability returned here is a claim about a real sensor or population.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import random


def finite_real(value, name):
    """Accept built-in int/float scalars, excluding bool and nonfinite values."""
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a built-in int or float, not bool")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside the finite float range") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def positive_integer(value, name, allow_zero=False):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not bool")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def probability(p):
    p = finite_real(p, "p")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    return p


def observations(values):
    if type(values) not in (list, tuple) or not values:
        raise ValueError("observations must be a nonempty flat list or tuple")
    if any(type(y) is not int or y not in (0, 1) for y in values):
        raise ValueError("each observation must be integer 0 or 1, not bool")
    return tuple(values)


def bernoulli_mass(y, p):
    y = observations([y])[0]
    p = probability(p)
    return p if y == 1 else 1.0 - p


def bernoulli_sample(n, marked_slots, total_slots, seed=17):
    """Sample rational p=marked_slots/total_slots by a finite uniform draw.

    The integer grid is the *declared model*, not a silent rounding of arbitrary p.
    Independent pseudo-random draws are a simulation of the model assumption.
    """
    n = positive_integer(n, "n", allow_zero=True)
    total_slots = positive_integer(total_slots, "total_slots")
    marked_slots = positive_integer(marked_slots, "marked_slots", allow_zero=True)
    if marked_slots > total_slots:
        raise ValueError("marked_slots must not exceed total_slots")
    if type(seed) is not int:
        raise TypeError("seed must be an integer, not bool")
    rng = random.Random(seed)
    return [int(rng.randrange(total_slots) < marked_slots) for _ in range(n)]


def bernoulli_likelihood(values, p):
    values = observations(values)
    p = probability(p)
    # Product is deliberately included to demonstrate floating-point underflow.
    return math.prod(p if y else 1.0 - p for y in values)


def bernoulli_log_likelihood(values, p):
    values = observations(values)
    p = probability(p)
    k = sum(values)
    n = len(values)
    # Test support before logarithms. Avoid 0 * log(0) and do not clip p.
    if p == 0.0:
        return 0.0 if k == 0 else -math.inf
    if p == 1.0:
        return 0.0 if k == n else -math.inf
    result = k * math.log(p) + (n - k) * math.log1p(-p)
    if not math.isfinite(result):
        raise ArithmeticError("finite-parameter log likelihood exceeded numeric range")
    return result


def finite_posterior(values, candidates, prior):
    """Bayes on a small *finite* parameter set, with explicit prior weights."""
    values = observations(values)
    if type(candidates) not in (list, tuple) or type(prior) not in (list, tuple):
        raise ValueError("candidates and prior must be flat lists or tuples")
    if not candidates or len(candidates) != len(prior):
        raise ValueError("candidate/prior lengths must match and be nonzero")
    ps = [probability(p) for p in candidates]
    if len(set(ps)) != len(ps):
        raise ValueError("candidate parameters must be distinct after float conversion")
    weights = [finite_real(w, "prior weight") for w in prior]
    if any(w < 0 for w in weights) or not math.isclose(math.fsum(weights), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("prior must be nonnegative and sum to one")
    scores = [(-math.inf if w == 0 else math.log(w) + bernoulli_log_likelihood(values, p))
              for p, w in zip(ps, weights)]
    shift = max(scores)
    if shift == -math.inf:
        raise ValueError("observation has zero probability under every positive-prior candidate")
    scaled = [math.exp(score - shift) for score in scores]
    total = math.fsum(scaled)
    return [weight / total for weight in scaled]


def gaussian_log_density(x, mu=0.0, sigma=1.0):
    x, mu, sigma = [finite_real(v, name) for v, name in ((x, "x"), (mu, "mu"), (sigma, "sigma"))]
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    difference = x - mu
    if not math.isfinite(difference):
        raise ArithmeticError("x-mu exceeded numeric range")
    z = difference / sigma
    if not math.isfinite(z):
        raise ArithmeticError("standardized coordinate exceeded numeric range")
    quadratic = (0.5 * z) * z
    result = -math.log(sigma) - 0.5 * math.log(2 * math.pi) - quadratic
    if not math.isfinite(result):
        raise ArithmeticError("Gaussian log density exceeded numeric range")
    return result


def gaussian_density(x, mu=0.0, sigma=1.0):
    log_density = gaussian_log_density(x, mu, sigma)
    try:
        density = math.exp(log_density)
    except OverflowError as exc:
        raise ArithmeticError("density exceeds float range; use log density") from exc
    if not math.isfinite(density):
        raise ArithmeticError("density exceeds float range; use log density")
    # A zero can be a numerical underflow, not mathematical zero density.
    return density


def gaussian_log_likelihood(values, mu=0.0, sigma=1.0):
    if type(values) not in (list, tuple) or not values:
        raise ValueError("Gaussian observations must be a nonempty flat list or tuple")
    result = math.fsum(gaussian_log_density(x, mu, sigma) for x in values)
    if not math.isfinite(result):
        raise ArithmeticError("Gaussian log likelihood exceeded numeric range")
    return result


def midpoint_integral(function, a, b, panels):
    """Finite interval midpoint quadrature; it is not a normalization proof."""
    a, b = finite_real(a, "a"), finite_real(b, "b")
    panels = positive_integer(panels, "panels")
    if panels > 1_000_000:
        raise ValueError("panels exceeds the teaching implementation limit")
    if not a < b:
        raise ValueError("require a < b")
    width = b - a
    if not math.isfinite(width):
        raise ArithmeticError("interval width exceeded numeric range")
    h = width / panels
    if h <= 0 or a + h == a or b - h == b:
        raise ArithmeticError("grid spacing cannot be resolved")
    terms = []
    previous = a
    for i in range(panels):
        x = a + (i + 0.5) * h
        if not a < x < b or x <= previous:
            raise ArithmeticError("midpoint grid cannot be resolved")
        value = finite_real(function(x), "integrand output")
        term = value * h
        if not math.isfinite(term):
            raise ArithmeticError("integral term exceeded numeric range")
        terms.append(term)
        previous = x
    try:
        total = math.fsum(terms)
    except OverflowError as exc:
        raise ArithmeticError("integral sum exceeded numeric range") from exc
    if not math.isfinite(total):
        raise ArithmeticError("integral sum exceeded numeric range")
    return total


def load_binary_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["trial", "y"]:
            raise ValueError("expected exact unique CSV header trial,y")
        rows = list(reader)
    if not rows or any(set(row) != {"trial", "y"} for row in rows):
        raise ValueError("expected nonempty CSV with trial,y columns")
    if [row["trial"] for row in rows] != [str(i + 1) for i in range(len(rows))]:
        raise ValueError("trial must be consecutive 1-based identifiers")
    if any(row["y"] not in ("0", "1") for row in rows):
        raise ValueError("CSV labels must be exact 0 or 1 strings")
    return list(observations([int(row["y"]) for row in rows]))


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(output_dir=None):
    base = Path(__file__).resolve().parent
    out = Path(output_dir) if output_dir is not None else base / "outputs"
    observed = load_binary_csv(base / "data" / "observations.csv")
    out.mkdir(parents=True, exist_ok=True)
    generated = bernoulli_sample(12, 3, 4, seed=17)
    rows = []
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        log_value = bernoulli_log_likelihood(observed, p)
        rows.append({"p": p, "likelihood": bernoulli_likelihood(observed, p),
                     "log_likelihood": "-inf" if log_value == -math.inf else log_value})
    write_csv(out / "bernoulli.csv", list(rows[0]), rows)
    integration_rows = []
    for bound in (1, 2, 4, 6):
        for panels in (20, 200, 2000):
            value = midpoint_integral(gaussian_density, -bound, bound, panels)
            integration_rows.append({"bound": bound, "panels": panels, "area": value})
    write_csv(out / "gaussian_integrals.csv", list(integration_rows[0]), integration_rows)
    failures = []
    cases = [
        ("invalid_probability", lambda: bernoulli_log_likelihood(observed, 1.01)),
        ("boolean_label", lambda: bernoulli_log_likelihood([True, 0], 0.5)),
        ("empty_sample", lambda: bernoulli_log_likelihood([], 0.5)),
        ("zero_scale", lambda: gaussian_density(0, 0, 0)),
        ("numeric_range", lambda: gaussian_log_density(1e308, -1e308, 1)),
        ("unresolvable_grid", lambda: midpoint_integral(lambda x: 1.0, 1e16, 1e16 + 4, 100))]
    for name, call in cases:
        try:
            call()
        except (ValueError, TypeError, ArithmeticError) as exc:
            failures.append({"case": name, "status": "rejected", "exception": type(exc).__name__})
        else:
            raise AssertionError(f"expected rejection: {name}")
    posterior = finite_posterior(observed, [0.25, 0.75], [0.9, 0.1])
    summary = {
        "data_kind": "original synthetic teaching data",
        "observed": observed, "n": len(observed), "successes": sum(observed),
        "generated_seed17_p3of4": generated,
        "long_sample_n": 2000, "long_product": bernoulli_likelihood([1] * 2000, 0.5),
        "long_log_likelihood": bernoulli_log_likelihood([1] * 2000, 0.5),
        "posterior_prior_9to1": posterior,
        "peak_sigma_0_1": gaussian_density(0, 0, 0.1),
        "area_minus0_1_to0_1_sigma0_1": midpoint_integral(lambda x: gaussian_density(x, 0, 0.1), -0.1, 0.1, 2000),
        "log_density_tail40": gaussian_log_density(40), "density_tail40": gaussian_density(40),
        "narrow_peak_coarse_area": midpoint_integral(lambda x: gaussian_density(x, 0, 0.01), -1, 1, 20),
        "narrow_peak_fine_area": midpoint_integral(lambda x: gaussian_density(x, 0, 0.01), -1, 1, 2000),
        "failures": failures}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run_experiment(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
