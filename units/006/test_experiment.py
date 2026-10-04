"""Replay, independence-of-test-labels, identity and failure tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import subprocess
import sys
import experiment as lab


def verify_artifacts(output):
    output=Path(output)
    if not (output/"run_status.json").exists():
        output=lab.latest_attempt(output)
    if json.loads((output/"run_status.json").read_text())["state"]!="complete":
        raise ValueError("The selected attempt did not complete")
    manifest=json.loads((output/"run_manifest.json").read_text())
    for name,expected in manifest["artifacts"].items():
        if lab.digest(output/name)!=expected:
            raise ValueError("Recorded artifact digest mismatch: "+name)
    return True


def run_tests():
    base=Path(__file__).resolve().parent
    frozen,trials,inputs,label_path=lab.prepare()
    primary=trials[0]
    expected_ids=["B3","D3","C1","B2","B4","A1","D2","A4"]
    assert primary["selected_train_ids"]==expected_ids
    assert [t["train_mae"]["affine"] for t in trials]==[0.25,0.625,0.5]
    assert [t["models"]["constant_mean"]["value"] for t in trials]==[6,5.375,6.5]
    assert [r["train_mae"] for r in primary["candidate_scores"]]==[1,0.25,1]
    assert primary["train_mae"]["constant_mean"]==2.25
    assert frozen["test_predictions"]==[3,5,7,9]
    assert frozen["selected_model"]=={"name":"affine","coefficient":2,"bias":1}
    groups=["exact selected IDs, candidate scores and hand-derived parameters"]
    for trial in trials:
        assert trial["validation_mae"]=={"affine":0,"constant_mean":2}
    assert lab.mae([5,7,9,11],[3,5,7,9])==2
    groups.append("all declared descriptive seeds and final hand-derived score")
    with TemporaryDirectory() as tmp:
        temp=Path(tmp)
        first=temp/"one";second=temp/"two"
        result1=lab.run(first);result2=lab.run(second)
        assert result1==result2
        first_attempt=lab.latest_attempt(first);second_attempt=lab.latest_attempt(second)
        for file in first_attempt.iterdir():assert file.read_bytes()==(second_attempt/file.name).read_bytes()
        assert verify_artifacts(first)
        groups.append("two complete reruns produce identical artifacts")
        target=first_attempt/"final_result.json";target.write_bytes(target.read_bytes()+b"\n")
        try:verify_artifacts(first)
        except ValueError:pass
        else:raise AssertionError("Modified result accepted without detecting changed bytes")
        groups.append("one-byte artifact change invalidates recorded digest")
        data=temp/"data";shutil.copytree(base/"data",data)
        labels=data/"test_labels.csv";labels.unlink()
        without_labels,_,_,_=lab.prepare(data_dir=data)
        assert without_labels==frozen
        try:lab.run(temp/"missing_labels",data_dir=data)
        except FileNotFoundError:pass
        else:raise AssertionError("Missing test labels did not stop evaluation")
        failed_attempt=lab.latest_attempt(temp/"missing_labels")
        assert (failed_attempt/"frozen_plan_and_predictions.json").exists()
        assert not (failed_attempt/"final_result.json").exists()
        assert json.loads((failed_attempt/"run_status.json").read_text())["state"]=="failed"
        groups.append("test labels physically absent during fitting and model selection")
        labels.write_text("id,y\nF1,100\nF2,100\nF3,100\nF4,100\n")
        changed_labels,_,_,_=lab.prepare(data_dir=data)
        assert changed_labels==frozen
        groups.append("changing test labels does not affect frozen choices or predictions")
        val=data/"validation.csv";original=val.read_text();val.write_text(original.replace(",E,",",A,"))
        try:lab.prepare(data_dir=data)
        except ValueError:pass
        else:raise AssertionError("Overlapping device accepted")
        val.write_text(original)
        groups.append("entity overlap rejected")
        config=json.loads((base/"config.json").read_text());config["train_sample_count"]=17
        bad_config=temp/"bad_config.json";lab.write_json(bad_config,config)
        try:lab.prepare(config_path=bad_config,data_dir=data)
        except ValueError:pass
        else:raise AssertionError("Oversized sample accepted")
        groups.append("impossible sample-count configuration rejected")
        # Regression: successful run, then changed plan + missing labels in SAME collection.
        mixed=temp/"repeated_collection"
        lab.run(mixed)
        old_attempt=lab.latest_attempt(mixed)
        old_bytes={p.name:p.read_bytes() for p in old_attempt.iterdir()}
        config["train_sample_count"]=8;config["candidate_biases"]=[0]
        lab.write_json(bad_config,config)
        labels.unlink()
        try:lab.run(mixed,config_path=bad_config,data_dir=data)
        except FileNotFoundError:pass
        else:raise AssertionError("Missing labels should fail the second attempt")
        newest=lab.latest_attempt(mixed)
        assert newest!=old_attempt
        assert {p.name:p.read_bytes() for p in old_attempt.iterdir()}==old_bytes
        assert json.loads((newest/"run_status.json").read_text())["state"]=="failed"
        assert json.loads((newest/"frozen_plan_and_predictions.json").read_text())["selected_model"]["bias"]==0
        assert not (newest/"final_result.json").exists()
        assert not (newest/"run_manifest.json").exists()
        try:verify_artifacts(mixed)
        except ValueError:pass
        else:raise AssertionError("Latest failed attempt was mistaken for an older success")
        assert verify_artifacts(old_attempt)
        groups.append("success then changed-plan failure in same collection preserves old run and marks new failure")
        script=base/"experiment.py"
        driver="import importlib.util; s=importlib.util.spec_from_file_location('fresh',"+repr(str(script))+"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); import json; print(json.dumps(m.run('fresh_outputs')))"
        run=subprocess.run([sys.executable,"-c",driver],cwd=temp,capture_output=True,text=True,check=True)
        assert json.loads(run.stdout)==result2
        assert verify_artifacts(temp/"fresh_outputs")
        groups.append("fresh process from unrelated directory reproduces final result")
    for actual,predicted in [([],[]),([1],[]),([float("nan")],[1])]:
        try:lab.mae(actual,predicted)
        except ValueError:pass
        else:raise AssertionError("Invalid metric data accepted")
    groups.append("empty, mismatched and nonfinite metric inputs rejected")
    return {"status":"passed","test_groups":groups}


if __name__=="__main__":
    print(json.dumps(run_tests(),indent=2))
