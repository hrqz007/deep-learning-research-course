"""Small inspectable experiment with named records, functions and model objects.

All data are synthetic. This module is safe to import: it performs no file I/O
until a function is called. Public teaching labels do not form a blind test.
"""
from pathlib import Path
import csv
import json
import math
import platform
import sys


def require_number(value):
    """Accept finite int/float measurements, rejecting bool and text."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("A finite numeric measurement is required")
    if not math.isfinite(value):
        raise ValueError("Measurements must be finite")
    return value


def mean_absolute_error(actual, predicted):
    """Equal-length nonempty numeric lists -> scalar mean absolute error."""
    if len(actual) == 0:
        raise ValueError("Cannot average empty data")
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths differ")
    total = 0.0
    for index in range(len(actual)):
        y = require_number(actual[index])
        prediction = require_number(predicted[index])
        total = total + abs(prediction - y)
    return total / len(actual)


def read_csv_records(path, columns):
    """Read exact CSV schema; preserve named IDs and convert other columns."""
    records = []
    seen_ids = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError("CSV header does not match the declared schema")
        row_number = 1
        for raw in reader:
            row_number = row_number + 1
            if len(raw) != len(columns) or None in raw:
                raise ValueError("Extra CSV fields at row " + str(row_number))
            for column in columns:
                if raw[column] is None:
                    raise ValueError("Missing CSV field at row " + str(row_number))
            identifier = raw["id"].strip()
            if identifier == "" or identifier in seen_ids:
                raise ValueError("Missing or repeated record ID at row " + str(row_number))
            seen_ids[identifier] = True
            record = {"id": identifier}
            for column in columns:
                if column != "id":
                    try:
                        value = float(raw[column])
                        require_number(value)
                    except ValueError as error:
                        message = "Invalid numeric field " + column + " at row " + str(row_number)
                        raise ValueError(message) from error
                    record[column] = value
            records.append(record)
    if len(records) == 0:
        raise ValueError("CSV contains no observations")
    return records


def field_values(records, field):
    result = []
    for record in records:
        result.append(record[field])
    return result


def labels_in_input_order(inputs, labels):
    mapping = {}
    for record in labels:
        if record["id"] in mapping:
            raise ValueError("Duplicate label identifier")
        mapping[record["id"]] = require_number(record["y"])
    if len(inputs) != len(mapping):
        raise ValueError("Input and label counts differ")
    seen_inputs = {}
    result = []
    for record in inputs:
        identifier = record["id"]
        if identifier in seen_inputs or identifier not in mapping:
            raise ValueError("Input identifiers are duplicated or unmatched")
        seen_inputs[identifier] = True
        result.append(mapping[identifier])
    return result


class PredictionRule:
    def predict_one(self, x):
        raise NotImplementedError("Provide a prediction rule")

    def predict(self, inputs):
        result = []
        for x in inputs:
            require_number(x)
            result.append(require_number(self.predict_one(x)))
        return result


class AffineRule(PredictionRule):
    def __init__(self, bias):
        self.bias = require_number(bias)

    def predict_one(self, x):
        return 2 * require_number(x) + self.bias


class LookupRule(PredictionRule):
    def __init__(self, training, fallback=0):
        self.fallback = require_number(fallback)
        self.memory = {}
        for record in training:
            x = require_number(record["x"])
            y = require_number(record["y"])
            if x in self.memory:
                raise ValueError("This lookup example requires unique training inputs")
            self.memory[x] = y

    def predict_one(self, x):
        require_number(x)
        if x in self.memory:
            return self.memory[x]
        return self.fallback


def choose_bias(training):
    """Fixed ascending candidates; strict improvement retains smaller ties."""
    inputs = field_values(training, "x")
    actual = field_values(training, "y")
    best_bias = 0
    best_error = None
    scores = []
    for bias in [0, 1, 2]:
        rule = AffineRule(bias)
        error = mean_absolute_error(actual, rule.predict(inputs))
        scores.append({"bias": bias, "train_mae": error})
        if best_error is None or error < best_error:
            best_bias = bias
            best_error = error
    return {"bias": best_bias, "scores": scores}


def run(output_dir=None):
    base = Path(__file__).resolve().parent
    if output_dir is None:
        output_dir = base / "outputs"
    output_dir = Path(output_dir)
    training = read_csv_records(base / "data/train.csv", ["id", "x", "y"])
    inputs = read_csv_records(base / "data/test_inputs.csv", ["id", "x"])
    selection = choose_bias(training)
    rule_a = AffineRule(selection["bias"])
    rule_b = LookupRule(training)
    train_x = field_values(training, "x")
    train_y = field_values(training, "y")
    test_x = field_values(inputs, "x")
    a_predictions = rule_a.predict(test_x)
    b_predictions = rule_b.predict(test_x)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = {"selection": selection, "lookup_fallback": rule_b.fallback,
              "input_ids": field_values(inputs, "id"), "A": a_predictions, "B": b_predictions}
    (output_dir / "frozen_predictions.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    # Reveal labels after freezing predictions. Public files are not secure blind tests.
    actual = labels_in_input_order(inputs, read_csv_records(base / "data/test_labels.csv", ["id", "y"]))
    stress = labels_in_input_order(inputs, read_csv_records(base / "data/stress_labels.csv", ["id", "y"]))
    metrics = {"selected_bias": selection["bias"],
               "A_train_mae": mean_absolute_error(train_y, rule_a.predict(train_x)),
               "B_train_mae": mean_absolute_error(train_y, rule_b.predict(train_x)),
               "A_test_mae": mean_absolute_error(actual, a_predictions),
               "B_test_mae": mean_absolute_error(actual, b_predictions),
               "A_stress_mae": mean_absolute_error(stress, a_predictions)}
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y", "A", "B", "A_absolute_error", "B_absolute_error"])
        for index in range(len(inputs)):
            writer.writerow([inputs[index]["id"], test_x[index], actual[index], a_predictions[index], b_predictions[index],
                             abs(a_predictions[index] - actual[index]), abs(b_predictions[index] - actual[index])])
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    versions = {"python": sys.version.split()[0], "platform": platform.system(), "dependencies": "Python standard library"}
    (output_dir / "environment.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
