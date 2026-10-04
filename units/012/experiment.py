"""Two affine layers with square activation: explicit JVP/VJP, no AD framework."""
from pathlib import Path
import argparse
import csv
import json
import math
import platform
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
PARAMETERS = ("W1", "b1", "W2", "b2")


def reject_boolean_leaves(value):
    if isinstance(value, (bool, np.bool_)) or (isinstance(value, np.ndarray) and value.dtype.kind == "b"):
        raise TypeError("Boolean data are not supported")
    if isinstance(value, (list, tuple)):
        for child in value:
            reject_boolean_leaves(child)


def matrix(value, name):
    reject_boolean_leaves(value)
    raw = np.asarray(value)
    if raw.ndim != 2 or 0 in raw.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        out = raw.astype(np.float64, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite after float64 conversion")
    return out


def finite(value):
    if not np.isfinite(value).all():
        raise ArithmeticError("A calculation exceeded the supported float64 range")
    return value


def validated(X, params):
    X = matrix(X, "X")
    if set(params) != set(PARAMETERS):
        raise ValueError("Parameters must contain exactly W1, b1, W2, b2")
    p = {k: matrix(params[k], k) for k in PARAMETERS}
    n, d = X.shape
    h = p["W1"].shape[1]
    c = p["W2"].shape[1]
    expected = {"W1": (d, h), "b1": (1, h), "W2": (h, c), "b2": (1, c)}
    if any(p[k].shape != expected[k] for k in PARAMETERS):
        raise ValueError("Incompatible parameter shapes; biases require explicit row axes")
    return X, p


def forward(X, params):
    X, p = validated(X, params)
    with np.errstate(over="ignore", invalid="ignore"):
        A = finite(X @ p["W1"] + p["b1"])
        H = finite(A * A)
        P = finite(H @ p["W2"] + p["b2"])
    return {"X": X, **p, "A": A, "H": H, "P": P}


def loss(X, params, Y):
    v = forward(X, params)
    Y = matrix(Y, "Y")
    if Y.shape != v["P"].shape:
        raise ValueError("Y must exactly match prediction shape; no label broadcasting")
    with np.errstate(over="ignore", invalid="ignore"):
        E = finite(v["P"] - Y)
        value = finite(np.sum(finite(E * E)) / (2 * len(v["X"])))
    return float(value)


def vjp(X, params, output_weights):
    """Pull back a fixed output weighting, with shapes matching primal arrays."""
    v = forward(X, params)
    G = matrix(output_weights, "output_weights")
    if G.shape != v["P"].shape:
        raise ValueError("Output weights must exactly match P")
    with np.errstate(over="ignore", invalid="ignore"):
        GW2 = finite(v["H"].T @ G)
        Gb2 = finite(G.sum(axis=0, keepdims=True))
        GH = finite(G @ v["W2"].T)
        GA = finite(GH * finite(2 * v["A"]))
        GW1 = finite(v["X"].T @ GA)
        Gb1 = finite(GA.sum(axis=0, keepdims=True))
        GX = finite(GA @ v["W1"].T)
    return {"W1": GW1, "b1": Gb1, "W2": GW2, "b2": Gb2, "X": GX, "H": GH, "A": GA, "P": G}


def gradients(X, params, Y):
    v = forward(X, params)
    Y = matrix(Y, "Y")
    if Y.shape != v["P"].shape:
        raise ValueError("Y must exactly match P")
    with np.errstate(over="ignore", invalid="ignore"):
        G = finite((v["P"] - Y) / len(v["X"]))
    return vjp(X, params, G)


def jvp(X, params, directions):
    """Push parameter directions forward while X remains fixed."""
    v = forward(X, params)
    if set(directions) != set(PARAMETERS):
        raise ValueError("Directions must have the four parameter names")
    d = {k: matrix(directions[k], k) for k in PARAMETERS}
    if any(d[k].shape != v[k].shape for k in PARAMETERS):
        raise ValueError("Each direction must match its parameter shape")
    with np.errstate(over="ignore", invalid="ignore"):
        dA = finite(v["X"] @ d["W1"] + d["b1"])
        dH = finite(finite(2 * v["A"]) * dA)
        dP = finite(dH @ v["W2"] + v["H"] @ d["W2"] + d["b2"])
    return dP


def flatten(arrays):
    return np.concatenate([arrays[k].ravel(order="C") for k in PARAMETERS])


def parameter_jacobian(X, params):
    """Explicit index derivatives, only for small teaching examples."""
    v = forward(X, params)
    n, c = v["P"].shape
    d, h = v["W1"].shape
    count = sum(v[k].size for k in PARAMETERS)
    if n * c * count > 100000:
        raise ValueError("Explicit teaching Jacobian would exceed 100000 entries")
    result = np.zeros((n * c, count), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(n):
            for o in range(c):
                row = []
                for j in range(d):
                    for k in range(h):
                        row.append(v["X"][i, j] * 2 * v["A"][i, k] * v["W2"][k, o])
                row.extend(2 * v["A"][i, k] * v["W2"][k, o] for k in range(h))
                row.extend(v["H"][i, k] if q == o else 0 for k in range(h) for q in range(c))
                row.extend(1 if q == o else 0 for q in range(c))
                result[i*c+o] = finite(np.asarray(row, dtype=np.float64))
    return result


def positive_step(value):
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise TypeError("h must be a real scalar")
    h = float(raw)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    if not math.isfinite(2*h):
        raise ArithmeticError("2h exceeds float64")
    return h


def central_gradients(X, params, Y, h):
    X, p = validated(X, params)
    h = positive_step(h)
    answer = {}
    for name in (*PARAMETERS, "X"):
        values = X if name == "X" else p[name]
        result = np.empty_like(values)
        for idx in np.ndindex(values.shape):
            plus, minus = values.copy(), values.copy()
            with np.errstate(over="ignore", invalid="ignore"):
                plus[idx] += h; minus[idx] -= h
            finite(plus); finite(minus)
            if plus[idx] == values[idx] or minus[idx] == values[idx]:
                raise ArithmeticError("Unresolvable input step")
            if name == "X":
                a, b = loss(plus, p, Y), loss(minus, p, Y)
            else:
                a = loss(X, {**p, name: plus}, Y)
                b = loss(X, {**p, name: minus}, Y)
            result[idx] = finite((a-b)/(2*h))
        answer[name] = finite(result)
    return answer


def load_config():
    cfg = json.loads((BASE/"data/network.json").read_text(encoding="utf-8"))
    X, params = validated(cfg["X"], cfg["parameters"])
    return cfg, X, params, matrix(cfg["Y"], "Y")


def run(output_dir=None):
    cfg, X, p, Y = load_config()
    v, g = forward(X, p), gradients(X, p, Y)
    J = parameter_jacobian(X, p)
    directions = {k: np.ones_like(p[k]) / 10 for k in PARAMETERS}
    pushed = jvp(X, p, directions)
    weights = np.arange(1, v["P"].size+1, dtype=float).reshape(v["P"].shape)
    pulled = vjp(X, p, weights)
    lhs = float(np.sum(weights * pushed))
    rhs = float(flatten(pulled) @ flatten(directions))
    rows = []
    for h in cfg["difference_steps"]:
        try:
            num = central_gradients(X, p, Y, h)
            error = max(float(np.max(np.abs(num[k]-g[k]))) for k in (*PARAMETERS,"X"))
            rows.append({"h": h, "max_absolute_error": error, "status": "computed"})
        except ArithmeticError as error:
            rows.append({"h": h, "max_absolute_error": None, "status": str(error)})
    result = {"loss": loss(X,p,Y), "forward": {k:v[k].tolist() for k in ("A","H","P")},
              "gradients": {k:g[k].tolist() for k in (*PARAMETERS,"X","A","H","P")},
              "parameter_order": list(PARAMETERS), "flatten_order": "row-major C",
              "jacobian": J.tolist(), "jacobian_shape": list(J.shape),
              "jvp_all_parameter_directions_0_1": pushed.tolist(),
              "adjoint_identity": {"lhs":lhs,"rhs":rhs},
              "vjp_full_jacobian_max_error":float(np.max(np.abs(g["P"].ravel() @ J-flatten(g))))}
    output = Path(output_dir) if output_dir is not None else BASE/"outputs"
    output.mkdir(parents=True, exist_ok=True)
    (output/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (output/"difference_scan.csv").open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['h','max_absolute_error','status']);writer.writeheader();writer.writerows(rows)
    (output/"environment.json").write_text(json.dumps({"python":sys.version.split()[0],"numpy":np.__version__,"platform":platform.system(),"device":"CPU","network_used":False},indent=2)+'\n')
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    print(json.dumps(run(args.output),ensure_ascii=False,indent=2,allow_nan=False))
