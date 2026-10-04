"""Original CPU NumPy example: explicit axes prevent silent pairwise scoring."""
from pathlib import Path
import csv
import json
import platform
import sys
import numpy as np


def load_data():
    path = Path(__file__).resolve().parent / "data/calibration.csv"
    identifiers, inputs, labels = [], [], []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "knob_1", "knob_2", "label"]:
            raise ValueError("Unexpected data schema")
        for row in reader:
            identifiers.append(row["id"])
            inputs.append([float(row["knob_1"]), float(row["knob_2"])])
            labels.append(float(row["label"]))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate sample identifiers")
    return identifiers, np.array(inputs, dtype=np.float64), np.array(labels, dtype=np.float64)


def validate_features(X):
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array")
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] == 0:
        raise ValueError("X must have shape (N, 2), with N > 0")
    if not (np.issubdtype(X.dtype, np.integer) or np.issubdtype(X.dtype, np.floating)):
        raise TypeError("X must contain real numerical values")
    if not np.isfinite(X).all():
        raise ValueError("X must contain finite values")


def loop_predict(X, weights, bias):
    validate_features(X)
    result = []
    for row in X.tolist():
        result.append(row[0] * float(weights[0]) + row[1] * float(weights[1]) + bias)
    return np.array(result, dtype=np.float64)


def array_predict(X, weights, bias):
    validate_features(X)
    if not isinstance(weights, np.ndarray) or weights.shape != (2,):
        raise ValueError("Weights must have shape (2,)")
    if not (np.issubdtype(weights.dtype, np.integer) or np.issubdtype(weights.dtype, np.floating)):
        raise TypeError("Weights must contain real values")
    if not np.isscalar(bias) or not isinstance(bias, (int, float, np.integer, np.floating)):
        raise TypeError("Bias must be a real scalar")
    if not np.isfinite(weights).all() or not np.isfinite(bias):
        raise ValueError("Parameters must be finite")
    weighted = X * weights
    combined = weighted.sum(axis=1)
    predictions = combined + bias
    if predictions.shape != (X.shape[0],):
        raise ValueError("Unexpected prediction shape")
    return weighted, combined, predictions


def strict_mae(labels, predictions):
    if not isinstance(labels, np.ndarray) or not isinstance(predictions, np.ndarray):
        raise TypeError("Both arguments must be NumPy arrays")
    if labels.ndim != 1 or predictions.ndim != 1 or labels.shape != predictions.shape:
        raise ValueError("This task requires matching one-dimensional shapes (N,)")
    if labels.size == 0:
        raise ValueError("Cannot score empty data")
    for array in [labels, predictions]:
        if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
            raise TypeError("This task requires real numerical values")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("Values must be finite")
    return float(np.abs(predictions - labels).mean())


def channel_marker():
    """Unique integer encodes sample/channel/row/column in separate places."""
    nchw = np.zeros((2, 3, 2, 2), dtype=np.int64)
    for n in range(2):
        for c in range(3):
            for h in range(2):
                for w in range(2):
                    nchw[n, c, h, w] = 1000 * n + 100 * c + 10 * h + w
    nhwc = nchw.transpose(0, 2, 3, 1)
    return nchw, nhwc


def run(output_dir=None):
    base = Path(__file__).resolve().parent
    output = Path(output_dir) if output_dir is not None else base / "outputs"
    identifiers, X, y = load_data()
    weights = np.array([2.0, 0.1], dtype=np.float64)
    weighted, combined, predictions = array_predict(X, weights, 1.0)
    loop = loop_predict(X, weights, 1.0)
    if loop.shape != predictions.shape:
        raise AssertionError("Loop/array shapes differ")
    np.testing.assert_allclose(loop, predictions, rtol=0, atol=1e-12)
    errors = predictions - y
    wrong_errors = predictions.reshape(3, 1) - y
    W = np.array([[2, -1], [0.1, 0.2]], dtype=np.float64)
    multi_output = X @ W + np.array([1, -1], dtype=np.float64)
    nchw, nhwc = channel_marker()
    arrays = {"X": X, "weights": weights, "weighted": weighted, "combined": combined,
              "predictions": predictions, "labels": y, "errors": errors,
              "pairwise_errors_deliberate_bug": wrong_errors, "multi_output": multi_output,
              "reshaped": X.reshape(2, 3), "transposed": X.T,
              "image_NCHW": nchw, "image_NHWC": nhwc}
    ledger = {}
    for name, array in arrays.items():
        ledger[name] = {"shape": list(array.shape), "dtype": str(array.dtype), "values": array.tolist()}
    metrics = {"correct_mae": strict_mae(y, predictions),
               "wrong_pairwise_mean_absolute_error": float(np.abs(wrong_errors).mean()),
               "weighted_axis_0_sum": weighted.sum(axis=0).tolist(),
               "weighted_axis_1_sum": weighted.sum(axis=1).tolist(),
               "weighted_all_mean": float(weighted.mean()),
               "sample_count": int(X.shape[0])}
    output.mkdir(parents=True, exist_ok=True)
    (output / "array_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "prediction", "absolute_error"])
        for i in range(len(identifiers)):
            writer.writerow([identifiers[i], float(y[i]), float(predictions[i]), float(abs(errors[i]))])
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.system()}
    (output / "environment.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
