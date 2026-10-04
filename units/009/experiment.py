"""DL009: batch-first row-vector affine maps; CPU NumPy, local synthetic data."""
from pathlib import Path
import argparse
import json
import platform
import sys
import numpy as np

HERE = Path(__file__).resolve().parent


def reject_boolean_leaves(value, name):
    """Check original list/tuple leaves before NumPy promotion loses bool type."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not contain boolean measurements")
    if isinstance(value, (list, tuple)):
        for item in value:
            reject_boolean_leaves(item, name)
    elif isinstance(value, np.ndarray) and value.dtype.kind == "b":
        raise TypeError(f"{name} must not contain a boolean array")


def finite_array(value, name):
    """Reject complex/object/string/bool values rather than silently coercing them."""
    reject_boolean_leaves(value, name)
    # An existing ndarray exposes its current dtype, not pre-conversion history.
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def matrix(value, name):
    result = finite_array(value, name)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must have two non-empty axes")
    return result


def bias_row(value, width, name="b"):
    result = finite_array(value, name)
    if result.shape == (width,):
        return result.reshape(1, width)
    if result.shape == (1, width):
        return result
    raise ValueError(f"{name} must have shape ({width},) or (1, {width})")


def linear(X, W):
    """X (N,D), W (D,H) -> (N,H); no automatic reshape or transpose."""
    X, W = matrix(X, "X"), matrix(W, "W")
    if X.shape[1] != W.shape[0]:
        raise ValueError(f"feature mismatch: X{X.shape}, W{W.shape}")
    with np.errstate(over="raise", invalid="raise"):
        result = X @ W
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("linear output is not finite")
    return result


def affine(X, W, b):
    """Shared output-feature bias only; never accepts per-sample biases."""
    Z = linear(X, W)
    b = bias_row(b, Z.shape[1])
    with np.errstate(over="raise", invalid="raise"):
        result = Z + b
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("affine output is not finite")
    return result


def compose_affine(W1, b1, W2, b2):
    """Row convention: f2(f1(X)) = X@(W1@W2) + b1@W2 + b2."""
    W1, W2 = matrix(W1, "W1"), matrix(W2, "W2")
    if W1.shape[1] != W2.shape[0]:
        raise ValueError("hidden width mismatch")
    b1 = bias_row(b1, W1.shape[1], "b1")
    b2 = bias_row(b2, W2.shape[1], "b2")
    return linear(W1, W2), affine(b1, W2, b2).reshape(-1)


def load_case(path=None):
    path = HERE / "data" / "plotter.json" if path is None else Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    arrays = {key: finite_array(data[key], key) for key in ("X", "W1", "b1", "W2", "b2")}
    if len(data["sample_ids"]) != arrays["X"].shape[0]:
        raise ValueError("sample_ids must match X rows")
    return data, arrays


def broadcast_evidence(b, N):
    b = finite_array(b, "b")
    if b.ndim != 1 or N < 1:
        raise ValueError("need a one-dimensional bias and positive N")
    view = np.broadcast_to(b, (N, len(b)))
    copied = np.tile(b, (N, 1))
    return {
        "same_values": bool(np.array_equal(view, copied)),
        "broadcast_shares_memory": bool(np.shares_memory(view, b)),
        "tile_shares_memory": bool(np.shares_memory(copied, b)),
        "broadcast_writeable": bool(view.flags.writeable),
        "bias_storage_bytes": int(b.nbytes),
        "expanded_logical_nbytes": int(view.nbytes),
        "explicit_tile_nbytes": int(copied.nbytes),
        "broadcast_strides": list(view.strides),
        "note": "view.nbytes describes logical elements, not new allocated storage; addition still allocates a result",
    }


def failure_examples(X, W1, b1):
    Z = linear(X, W1)
    wrong_bias = Z + b1[:, None]  # N=H=3: valid broadcasting, wrong meaning.
    wrong_transpose = X.T @ X   # Valid but feature-by-feature aggregation, not sample output.
    S = np.array([[2., 0.], [0., 1.]])
    T = np.array([[1., 1.], [0., 1.]])
    point = np.array([[1., 1.]])
    changed = W1.copy()
    changed[0, 1] += 0.25
    observed = linear(X, changed) - linear(X, W1)
    expected = np.zeros_like(observed)
    expected[:, 1] = 0.25 * X[:, 0]
    return {
        "wrong_bias": wrong_bias.tolist(),
        "correct_bias": affine(X, W1, b1).tolist(),
        "wrong_bias_max_abs_error": float(np.max(np.abs(wrong_bias - affine(X, W1, b1)))),
        "X_transpose_X": wrong_transpose.tolist(),
        "ST": (S @ T).tolist(), "TS": (T @ S).tolist(),
        "point_ST": (point @ S @ T).tolist(), "point_TS": (point @ T @ S).tolist(),
        "parameter_change": observed.tolist(), "parameter_change_expected": expected.tolist(),
        "transpose_is_not_inverse": (point @ S @ S.T).tolist(),
    }


def run(output=None):
    data, a = load_case()
    X, W1, b1, W2, b2 = (a[k] for k in ("X", "W1", "b1", "W2", "b2"))
    Z = linear(X, W1)
    U = affine(X, W1, b1)
    Y = affine(U, W2, b2)
    Wc, bc = compose_affine(W1, b1, W2, b2)
    collapsed = affine(X, Wc, bc)
    arrays = {"X": X, "W1": W1, "b1": b1, "Z": Z, "U": U,
              "W2": W2, "b2": b2, "Y": Y, "Wc": Wc, "bc": bc}
    report = {
        "sample_ids": data["sample_ids"],
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "arrays": {k: v.tolist() for k, v in arrays.items()},
        "composition_max_abs_error": float(np.max(np.abs(Y-collapsed))),
        "associativity_max_abs_error": float(np.max(np.abs((X@W1)@W2-X@(W1@W2)))),
        "origin_output_mm": affine(np.zeros((1, 2)), Wc, bc).tolist(),
        "basis_images_without_bias": linear(np.eye(2), Wc).tolist(),
        "broadcast": broadcast_evidence(b1, X.shape[0]),
        "failures": failure_examples(X, W1, b1),
    }
    if output is not None:
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        np.savetxt(output / "positions_mm.csv", Y, delimiter=",", header="horizontal_mm,vertical_mm", comments="", fmt="%.8g")
        environment = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
                       "computation": "CPU, float64; no network/API; deterministic fixed inputs"}
        (output / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": "passed", "Y_mm": result["arrays"]["Y"],
                      "Wc": result["arrays"]["Wc"], "bc_mm": result["arrays"]["bc"],
                      "composition_max_abs_error": result["composition_max_abs_error"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
