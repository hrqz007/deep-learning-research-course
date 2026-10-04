"""Deliberately faulty teaching examples. Never used by the real experiment."""
from pathlib import Path


def wrong_indices(values):
    result = []
    for index in range(1, len(values) + 1):
        result.append(values[index])
    return result


def unconverted_prediction(raw_x):
    return 2 * raw_x + 1


def wrong_working_directory(directory):
    return (Path(directory) / "train.csv").read_text(encoding="utf-8")


def wrong_denominator(actual, predicted):
    total = 0
    for index in range(len(actual)):
        total = total + abs(predicted[index] - actual[index])
    return total / 3
