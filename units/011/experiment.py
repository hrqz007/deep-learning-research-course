"""Original two-parameter calculus lab. CPU, local JSON, NumPy float64 only."""
from pathlib import Path
import argparse
import csv
import json
import math
import platform
import sys
import numpy as np

BASE = Path(__file__).resolve().parent


def reject_boolean_leaves(value):
    """Inspect original Python containers before NumPy numeric promotion."""
    if isinstance(value, (bool, np.bool_)) or (isinstance(value, np.ndarray) and value.dtype.kind == "b"):
        raise TypeError("Boolean coordinates are not supported")
    if isinstance(value, (list, tuple)):
        for child in value:
            reject_boolean_leaves(child)


def row2(value, name="point"):
    reject_boolean_leaves(value)
    raw = np.asarray(value)
    if raw.shape != (1, 2):
        raise ValueError(f"{name} must have shape (1, 2); got {raw.shape}")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers, not bool, complex or text")
    with np.errstate(over="ignore", invalid="ignore"):
        out = raw.astype(np.float64, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain finite float64 numbers")
    return out


def finite_scalar(value, name="value"):
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real numeric scalar")
    try:
        value = float(raw)
    except OverflowError as exc:
        raise ArithmeticError(f"{name} exceeds float64") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def checked_result(value):
    out = np.asarray(value)
    if not np.isfinite(out).all():
        raise ArithmeticError("Calculation exceeded the supported numeric range")
    return value


def forward(point):
    p = row2(point)
    x, y = map(float, p[0])
    a = checked_result(x * y)
    z = checked_result(a + x)
    loss = checked_result(z * z)
    return {"x": x, "y": y, "a": a, "z": z, "loss": loss}


def loss(point):
    return forward(point)["loss"]


def gradient(point):
    values = forward(point)
    x, y, z = values["x"], values["y"], values["z"]
    g = np.array([[2 * z * (y + 1), 2 * z * x]], dtype=np.float64)
    return checked_result(g)


def path_contributions(point):
    v = forward(point)
    direct = checked_result(2 * v["z"])
    via_a = checked_result(direct * v["y"])
    y_path = checked_result(direct * v["x"])
    return {"x_via_a": via_a, "x_direct": direct,
            "x_total": checked_result(via_a + direct), "y_total": y_path}


def central_gradient(function, point, h):
    p = row2(point)
    h = finite_scalar(h, "h")
    if h <= 0:
        raise ValueError("h must be positive")
    result = np.empty((1, 2), dtype=np.float64)
    for j in range(2):
        plus, minus = p.copy(), p.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            plus[0, j] += h
            minus[0, j] -= h
        if not np.isfinite(plus).all() or not np.isfinite(minus).all():
            raise ArithmeticError("Shifted input overflow")
        if plus[0, j] == p[0, j] or minus[0, j] == p[0, j]:
            raise ArithmeticError("Unresolvable input step")
        top = finite_scalar(function(plus), "function output")
        bottom = finite_scalar(function(minus), "function output")
        result[0, j] = checked_result(((top - bottom) / h) / 2)
    return checked_result(result)


def rate_along(point, velocity):
    g = gradient(point)
    v = row2(velocity, "velocity")
    with np.errstate(over="ignore", invalid="ignore"):
        rate = float((g @ v.T)[0, 0])
    return checked_result(rate)


def directional_derivative(point, direction):
    u = row2(direction, "direction")
    length = math.hypot(float(u[0, 0]), float(u[0, 1]))
    if not math.isfinite(length) or not math.isclose(length, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Direction must be unit length; normalize explicitly")
    return rate_along(point, u)


def curve_loss(t):
    t = finite_scalar(t, "t")
    return loss([[t, t + 1]])


def central_curve(t, h):
    t, h = finite_scalar(t, "t"), finite_scalar(h, "h")
    if h <= 0:
        raise ValueError("h must be positive")
    if not math.isfinite(t + h) or not math.isfinite(t - h):
        raise ArithmeticError("Shifted input overflow")
    if t + h == t or t - h == t:
        raise ArithmeticError("Unresolvable input step")
    return checked_result(((curve_loss(t + h) - curve_loss(t - h)) / h) / 2)


def linear_prediction(point, delta):
    p, d = row2(point), row2(delta, "delta")
    first_order = rate_along(p, d)
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = checked_result(p + d)
    exact = loss(shifted)
    estimate = checked_result(loss(p) + first_order)
    return {"delta": d.tolist(), "linear_change": first_order,
            "predicted_loss": estimate, "actual_loss": exact,
            "remainder": checked_result(exact - estimate)}


def gradient_step(point, eta):
    p = row2(point)
    eta = finite_scalar(eta, "eta")
    if eta <= 0:
        raise ValueError("eta must be positive")
    g = gradient(p)
    with np.errstate(over="ignore", invalid="ignore"):
        q = checked_result(p - eta * g)
    with np.errstate(over="ignore", invalid="ignore"):
        predicted = checked_result(-eta * float(np.sum(g * g)))
    return {"eta": eta, "new_point": q.tolist(), "old_loss": loss(p),
            "new_loss": loss(q), "linear_predicted_change": predicted}


def partials_not_enough(point):
    p = row2(point)
    x, y = map(float, p[0])
    denominator = checked_result(math.hypot(x, y))
    if denominator == 0:
        return 0.0
    # Multiplying after division prevents a needless x*y overflow.
    return checked_result((x / denominator) * y)


def run(output_dir=None):
    output = Path(output_dir) if output_dir is not None else BASE / "outputs"
    config = json.loads((BASE / "data/experiment_config.json").read_text(encoding="utf-8"))
    p = row2(config["point"])
    if not np.array_equal(p, [[2.0, 3.0]]):
        raise ValueError("This fixed teaching report requires point [[2,3]]; use core functions for other points")
    scans = []
    for h in config["difference_steps"]:
        try:
            numeric = central_gradient(loss, p, h)
            curve = central_curve(2.0, h)
            item = {"h": h, "gx": float(numeric[0, 0]), "gy": float(numeric[0, 1]),
                    "gradient_max_error": float(np.max(np.abs(numeric - gradient(p)))),
                    "curve_rate": curve, "curve_error": abs(curve - 96.0), "status": "computed"}
        except ArithmeticError as exc:
            item = {"h": h, "gx": None, "gy": None, "gradient_max_error": None,
                    "curve_rate": None, "curve_error": None, "status": str(exc)}
        scans.append(item)
    u = np.array([[3.0, 4.0]]) / 5.0
    tangent = np.array([[-1.0, 2.0]]) / math.sqrt(5)
    eps = 0.01
    report = {"forward": forward(p), "gradient_row": gradient(p).tolist(),
              "gradient_shape": list(gradient(p).shape), "path_contributions": path_contributions(p),
              "direction_3_4_unit_rate": directional_derivative(p, u),
              "velocity_3_4_rate": rate_along(p, [[3, 4]]),
              "tangent_rate": directional_derivative(p, tangent),
              "curve_at_t2_rate": rate_along(p, [[1, 1]]),
              "linear_checks": [linear_prediction(p, [[e, 2 * e]]) for e in config["linear_scales"]],
              "steps": [gradient_step(p, eta) for eta in config["learning_rates"]],
              "counterexample": {"central_partials_at_origin": central_gradient(partials_not_enough, [[0, 0]], eps).tolist(),
                                   "diagonal_right_quotient": partials_not_enough([[eps, eps]]) / eps,
                                   "diagonal_left_quotient": partials_not_enough([[-eps, -eps]]) / (-eps),
                                   "differentiable_at_origin": False}}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "difference_scan.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scans[0]))
        writer.writeheader(); writer.writerows(scans)
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    env = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.system(), "device": "CPU", "network_used": False}
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(BASE / "outputs"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2, allow_nan=False))
