"""Original synthetic calibration experiment. Python standard library only.

Teaching test labels are public. File separation illustrates a protocol; it is
not an access-control boundary and does not make this a genuine blind test.
"""
from pathlib import Path
import csv
import json
import math
import platform
import sys


def checked_numbers(values):
    result = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("Boolean values are not calibration measurements")
        try:
            number = float(value)
        except (ValueError, TypeError) as error:
            raise ValueError("All measurements must be numeric") from error
        if not math.isfinite(number):
            raise ValueError("All measurements must be finite")
        result.append(number)
    return result


def mae(actual, predicted):
    actual = checked_numbers(actual)
    predicted = checked_numbers(predicted)
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths differ")
    if not actual:
        raise ValueError("Mean error is undefined for empty data")
    return sum(abs(p - y) for y, p in zip(actual, predicted)) / len(actual)


def read_rows(path, columns):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError("Unexpected CSV columns")
        rows = list(reader)
    if not rows or any(set(row) != set(columns) for row in rows):
        raise ValueError("Empty or malformed CSV")
    ids = [row["id"] for row in rows]
    if any(not identifier for identifier in ids) or len(set(ids)) != len(ids):
        raise ValueError("Record identifiers must be nonempty and unique")
    for row in rows:
        for column in columns:
            if column != "id":
                row[column] = checked_numbers([row[column]])[0]
    return rows


def choose_bias(training, candidates=(0, 1, 2)):
    candidates = checked_numbers(candidates)
    if not training or not candidates:
        raise ValueError("Training records and candidates must be nonempty")
    scores = []
    for bias in candidates:
        predictions = [2 * row["x"] + bias for row in training]
        scores.append({"bias": bias, "train_mae": mae([r["y"] for r in training], predictions)})
    winner = min(scores, key=lambda row: (row["train_mae"], row["bias"]))
    return winner["bias"], scores


def lookup_predictions(training, inputs, fallback=0):
    xs = [row["x"] for row in training]
    if len(set(xs)) != len(xs):
        raise ValueError("This teaching lookup requires unique training inputs")
    memory = {row["x"]: row["y"] for row in training}
    return [memory.get(row["x"], fallback) for row in inputs]


def align_labels(inputs, labels):
    input_ids = [row["id"] for row in inputs]
    label_ids = [row["id"] for row in labels]
    if len(set(input_ids)) != len(input_ids) or len(set(label_ids)) != len(label_ids):
        raise ValueError("Duplicate identifiers")
    if set(input_ids) != set(label_ids):
        raise ValueError("Prediction and label identifiers differ")
    mapping = {row["id"]: row["y"] for row in labels}
    return [mapping[identifier] for identifier in input_ids]


def run_self_tests():
    assert mae([8, 8], [7, 9]) == 1
    cases = [([], []), ([1], []), ([float("nan")], [1]), ([1], [float("inf")]), ([True], [1])]
    for actual, predicted in cases:
        try:
            mae(actual, predicted)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid metric input was accepted")
    assert lookup_predictions([{"x": 1, "y": 3}], [{"x": 2}]) == [0]
    try:
        lookup_predictions([{"x": 1, "y": 3}, {"x": 1, "y": 4}], [{"x": 1}])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate training inputs accepted")
    try:
        align_labels([{"id": "E1"}], [{"id": "E2", "y": 1}])
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched labels accepted")
    assert align_labels([{"id": "E2"}, {"id": "E1"}], [{"id": "E1", "y": 4}, {"id": "E2", "y": 6}]) == [6, 4]
    bias, scores = choose_bias([{"x": 0, "y": 0.5}], [1, 0])
    assert bias == 0, "Tie must select the smallest bias"
    return {"status": "passed", "checks": 11}


def run(output_dir=None):
    base = Path(__file__).resolve().parent
    output = Path(output_dir) if output_dir is not None else base / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    training = read_rows(base / "data/train.csv", ["id", "x", "y"])
    inputs = read_rows(base / "data/test_inputs.csv", ["id", "x"])
    bias, candidates = choose_bias(training)
    a_test = [2 * row["x"] + bias for row in inputs]
    b_test = lookup_predictions(training, inputs)
    frozen = {"coefficient": 2, "bias": bias, "candidates": candidates, "lookup_fallback": 0,
              "predictions": [{"id": row["id"], "x": row["x"], "A": a, "B": b} for row, a, b in zip(inputs, a_test, b_test)]}
    (output / "frozen_predictions.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    # Only now reveal the public teaching labels. This is a pedagogical ordering.
    actual = align_labels(inputs, read_rows(base / "data/test_labels.csv", ["id", "y"]))
    stress = align_labels(inputs, read_rows(base / "data/stress_labels.csv", ["id", "y"]))
    train_y = [row["y"] for row in training]
    metrics = {"selected_bias": bias,
               "A_train_mae": mae(train_y, [2 * row["x"] + bias for row in training]),
               "B_train_mae": mae(train_y, lookup_predictions(training, training)),
               "A_test_mae": mae(actual, a_test), "B_test_mae": mae(actual, b_test),
               "A_stress_mae": mae(stress, a_test), "n_train": len(training), "n_test": len(inputs)}
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y", "A", "A_absolute_error", "B", "B_absolute_error", "stress_y", "A_stress_absolute_error"])
        for row, y, a, b, s in zip(inputs, actual, a_test, b_test, stress):
            writer.writerow([row["id"], row["x"], y, a, abs(a - y), b, abs(b - y), s, abs(a - s)])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "environment.json").write_text(json.dumps({"python": sys.version.split()[0], "platform": platform.system(), "dependencies": "Python standard library"}, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    run_self_tests()
    print(json.dumps(run(), indent=2))
