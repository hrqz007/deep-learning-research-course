"""Executable teaching tests. Run normally, without Python's -O option."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
import sys
import experiment as lab
import faults


def run_tests():
    records = []
    assert lab.mean_absolute_error([3, 5], [2, 6]) == 1
    assert lab.mean_absolute_error([8, 8], [7, 9]) == 1
    assert abs(lab.mean_absolute_error([3, 5, 8, 9, 11], [3, 5, 7, 9, 11]) - 0.2) < 1e-12
    records.append("three hand-calculated metric fixtures")
    for actual, predicted in [([], []), ([1], []), ([float("nan")], [1])]:
        try:
            lab.mean_absolute_error(actual, predicted)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid metric input accepted")
    try:
        lab.mean_absolute_error(["3"], [3])
    except TypeError:
        pass
    else:
        raise AssertionError("Text measurements accepted")
    records.append("empty, mismatched, nonfinite and text metric inputs rejected")
    first = lab.AffineRule(1)
    second = lab.AffineRule(2)
    assert first.predict_one(3) == 7 and second.predict_one(3) == 8
    second.bias = 4
    assert first.predict_one(3) == 7
    inputs = [1.5, 2.5]
    before = inputs.copy()
    assert first.predict(inputs) == [4, 6]
    assert inputs == before
    records.append("instance isolation, inherited method and nonmutation")
    try:
        lab.PredictionRule().predict([1])
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Unimplemented rule accepted")
    labels = [{"id": "E2", "y": 6}, {"id": "E1", "y": 4}]
    assert lab.labels_in_input_order([{"id": "E1"}, {"id": "E2"}], labels) == [4, 6]
    try:
        lab.labels_in_input_order([{"id": "E1"}, {"id": "E1"}], labels)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate input IDs accepted")
    records.append("base interface and ID alignment")
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for text in ["id,x,y\n", "id,x,y\nT01,1,3,extra\n", "id,x,y\nT01,1\n", "id,x,y\nT01,bad,3\n", "id,x,y\nT01,1,3\nT01,2,5\n", "id,x,y\nT01,inf,3\n"]:
            file = directory / "bad.csv"
            file.write_text(text, encoding="utf-8")
            try:
                lab.read_csv_records(file, ["id", "x", "y"])
            except ValueError:
                pass
            else:
                raise AssertionError("Malformed CSV accepted")
        records.append("six malformed CSV cases rejected")
        try:
            faults.wrong_indices([1, 2, 3])
        except IndexError:
            pass
        else:
            raise AssertionError("Index fault not reproduced")
        try:
            faults.unconverted_prediction("1.5")
        except TypeError:
            pass
        else:
            raise AssertionError("Type fault not reproduced")
        try:
            faults.wrong_working_directory(directory)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Path fault not reproduced")
        wrong = faults.wrong_denominator([3, 5], [2, 6])
        assert abs(wrong - 1) > 0.1
        records.append("three exception faults and a silent wrong denominator reproduced")
        metrics = lab.run(directory / "results")
        expected = {"selected_bias": 1, "A_train_mae": 0.2, "B_train_mae": 0, "A_test_mae": 0, "B_test_mae": 7, "A_stress_mae": 3}
        for key in expected:
            assert abs(metrics[key] - expected[key]) < 1e-12
        records.append("integration metrics match independent hand fixtures")
        source = Path(__file__).resolve().parent / "experiment.py"
        isolated = directory / "isolated"
        isolated.mkdir()
        isolated_source = isolated / "experiment.py"
        isolated_source.write_bytes(source.read_bytes())
        runner = "import importlib.util; s=importlib.util.spec_from_file_location('fresh_lab'," + repr(str(isolated_source)) + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('imported')"
        before_files = sorted(p.name for p in directory.iterdir())
        imported = subprocess.run([sys.executable, "-c", runner], cwd=directory, capture_output=True, text=True, check=True)
        assert imported.stdout.strip() == "imported"
        assert sorted(p.name for p in directory.iterdir()) == before_files
        assert not (isolated / "outputs").exists()
        records.append("fresh-process import needs no data and creates no result files")
        runner = runner.replace(repr(str(isolated_source)), repr(str(source)))
        runner = runner.replace("print('imported')", "import json; print(json.dumps(m.run('alternate_outputs')))")
        alternative = subprocess.run([sys.executable, "-c", runner], cwd=directory, capture_output=True, text=True, check=True)
        assert json.loads(alternative.stdout) == metrics
        records.append("same results from an unrelated working directory")
    return {"status": "passed", "test_groups": records}


if __name__ == "__main__":
    print(json.dumps(run_tests(), indent=2))
