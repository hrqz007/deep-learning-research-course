"""A fully recorded, tiny, descriptive calibration experiment; stdlib only."""
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
import random
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_records(path, fields):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError("Unexpected CSV schema")
        records = list(reader)
    if not records:
        raise ValueError("Data must not be empty")
    seen = set()
    for row in records:
        if None in row or any(row.get(key) is None for key in fields):
            raise ValueError("Malformed CSV record")
        if not row["id"] or row["id"] in seen:
            raise ValueError("Record IDs must be nonempty and unique")
        seen.add(row["id"])
        if "device" in fields and not row["device"]:
            raise ValueError("Device identifier is required")
        for key in ["x", "y"]:
            if key in fields:
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError("Measurement must be finite")
    return records


def mae(actual, predicted):
    if not actual or len(actual) != len(predicted):
        raise ValueError("Need equal, nonempty arrays")
    if not all(math.isfinite(v) for v in actual + predicted):
        raise ValueError("Metric requires finite numbers")
    return sum(abs(y-p) for y,p in zip(actual,predicted))/len(actual)


def validate_config(config, count):
    if config["protocol_version"] != 1 or config["metric"] != "mean_absolute_error":
        raise ValueError("Unsupported protocol or metric")
    if config["split_by"] != "device" or config["model_tie_break"] != "affine":
        raise ValueError("This exercise uses device splits and affine tie-break")
    if type(config["train_sample_count"]) is not int or not 1 <= config["train_sample_count"] <= count:
        raise ValueError("Training sample count is out of range")
    seeds=config["descriptive_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("Seeds must be unique integers")
    if config["primary_seed"] not in seeds:
        raise ValueError("Primary seed must be declared among descriptive seeds")
    biases=config["candidate_biases"]
    if not biases or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in biases):
        raise ValueError("Candidate biases must be finite and nonempty")
    if not math.isfinite(config["fixed_coefficient"]):
        raise ValueError("Coefficient must be finite")
    for name in config["data_files"].values():
        if Path(name).name != name:
            raise ValueError("Data files must be local filenames")


def predict(model, inputs):
    if model["name"] == "constant_mean":
        return [model["value"] for row in inputs]
    if model["name"] == "affine":
        return [model["coefficient"]*row["x"]+model["bias"] for row in inputs]
    raise ValueError("Unknown model name")


def fit_trial(training, validation, config, seed):
    rng=random.Random(seed)
    subset=rng.sample(training, config["train_sample_count"])
    ys=[r["y"] for r in subset]
    constant={"name":"constant_mean", "value":sum(ys)/len(ys)}
    scores=[]
    for bias in sorted(config["candidate_biases"]):
        model={"name":"affine", "coefficient":config["fixed_coefficient"], "bias":bias}
        scores.append({"bias":bias,"train_mae":mae(ys,predict(model,subset))})
    winner=min(scores,key=lambda r:(r["train_mae"],r["bias"]))
    affine={"name":"affine","coefficient":config["fixed_coefficient"],"bias":winner["bias"]}
    val_y=[r["y"] for r in validation]
    return {"seed":seed,"selected_train_ids":[r["id"] for r in subset],"candidate_scores":scores,
            "models":{"constant_mean":constant,"affine":affine},
            "train_mae":{"constant_mean":mae(ys,predict(constant,subset)),"affine":winner["train_mae"]},
            "validation_mae":{"constant_mean":mae(val_y,predict(constant,validation)),"affine":mae(val_y,predict(affine,validation))}}


def prepare(config_path=None, data_dir=None):
    """Prepare frozen predictions without opening test_labels.csv."""
    base=Path(__file__).resolve().parent
    config_path=Path(config_path) if config_path is not None else base/"config.json"
    data_dir=Path(data_dir) if data_dir is not None else base/"data"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    files=config["data_files"]
    for filename in files.values():
        if Path(filename).name != filename:
            raise ValueError("Data files must be local filenames")
    training=read_records(data_dir/files["train"],["id","device","x","y"])
    validation=read_records(data_dir/files["validation"],["id","device","x","y"])
    inputs=read_records(data_dir/files["test_inputs"],["id","device","x"])
    validate_config(config,len(training))
    split_devices=[set(r["device"] for r in rows) for rows in [training,validation,inputs]]
    if split_devices[0]&split_devices[1] or split_devices[0]&split_devices[2] or split_devices[1]&split_devices[2]:
        raise ValueError("Device overlap across splits")
    all_ids=[r["id"] for rows in [training,validation,inputs] for r in rows]
    if len(all_ids)!=len(set(all_ids)):
        raise ValueError("Sample ID overlap across splits")
    trials=[fit_trial(training,validation,config,seed) for seed in config["descriptive_seeds"]]
    primary=next(t for t in trials if t["seed"]==config["primary_seed"])
    selected="affine" if primary["validation_mae"]["affine"] <= primary["validation_mae"]["constant_mean"] else "constant_mean"
    model=primary["models"][selected]
    frozen={"config":config,"primary_seed":config["primary_seed"],"selected_model":model,
            "selection_basis":"Primary-seed validation MAE; affine wins ties",
            "test_input_ids":[r["id"] for r in inputs],"test_predictions":predict(model,inputs),
            "pretest_data_sha256":{role:digest(data_dir/files[role]) for role in ["train","validation","test_inputs"]},
            "config_sha256":digest(config_path),"source_sha256":digest(Path(__file__)),
            "split_devices":{name:sorted(group) for name,group in zip(["train","validation","test"],split_devices)}}
    return frozen,trials,inputs,data_dir/files["test_labels"]


def _run_attempt(output_dir, config_path=None, data_dir=None):
    base=Path(__file__).resolve().parent
    output=Path(output_dir) if output_dir is not None else base/"outputs"
    output.mkdir(parents=True,exist_ok=True)
    frozen,trials,inputs,label_path=prepare(config_path,data_dir)
    write_json(output/"frozen_plan_and_predictions.json",frozen)
    write_json(output/"all_descriptive_trials.json",trials)
    # Only after the model/decisions/predictions are frozen, reveal teaching test labels.
    labels=read_records(label_path,["id","y"])
    mapping={r["id"]:r["y"] for r in labels}
    if set(mapping)!=set(frozen["test_input_ids"]):
        raise ValueError("Test labels do not match input IDs")
    actual=[mapping[identifier] for identifier in frozen["test_input_ids"]]
    predictions=frozen["test_predictions"]
    result={"selected_model":frozen["selected_model"],"primary_seed":frozen["primary_seed"],
            "test_mae":mae(actual,predictions),"test_sample_count":len(actual),"test_device_count":len(frozen["split_devices"]["test"]),
            "scope":"Public synthetic descriptive exercise; not a blind or population-level finding"}
    write_json(output/"final_result.json",result)
    with (output/"test_predictions.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["id","device","x","actual","prediction","signed_error","absolute_error"])
        for row,y,p in zip(inputs,actual,predictions):writer.writerow([row["id"],row["device"],row["x"],y,p,p-y,abs(p-y)])
    manifest={"input_sha256":{**frozen["pretest_data_sha256"],"test_labels":digest(label_path)},
              "config_sha256":frozen["config_sha256"],"source_sha256":frozen["source_sha256"],
              "runtime":{"python":sys.version.split()[0],"platform":platform.system()},
              "artifacts":{name:digest(output/name) for name in ["frozen_plan_and_predictions.json","all_descriptive_trials.json","final_result.json","test_predictions.csv"]}}
    write_json(output/"run_manifest.json",manifest)
    log=["Read configuration and named input files","Verified unique IDs and disjoint devices","Fit all declared training-subset seeds; retained every trial","Selected model on primary-seed validation only","Saved frozen model and test predictions","Opened public test labels and saved row-level errors","Saved content digests and runtime versions"]
    (output/"run_log.txt").write_text("\n".join(str(i+1)+". "+line for i,line in enumerate(log))+"\n",encoding="utf-8")
    return result


def latest_attempt(collection):
    """Return newest attempt, even when it failed; never silently use old success."""
    candidates=[]
    for path in Path(collection).iterdir():
        if path.is_dir() and path.name.startswith("run-") and path.name[4:].isdigit():
            candidates.append(path)
    if not candidates:
        raise ValueError("No recorded run attempt")
    return max(candidates,key=lambda path:int(path.name[4:]))


def run(output_dir=None, config_path=None, data_dir=None):
    """Every attempt gets a new directory and explicit state; old runs stay intact."""
    base=Path(__file__).resolve().parent
    collection=Path(output_dir) if output_dir is not None else base/"outputs"
    collection.mkdir(parents=True,exist_ok=True)
    if (collection/"final_result.json").exists() or (collection/"frozen_plan_and_predictions.json").exists():
        raise ValueError("Legacy flat output layout found; choose a fresh run collection")
    number=1
    while True:
        attempt=collection/("run-"+str(number).zfill(4))
        try:
            attempt.mkdir()
            break
        except FileExistsError:
            number+=1
    write_json(attempt/"run_status.json",{"state":"running"})
    try:
        result=_run_attempt(attempt,config_path,data_dir)
    except Exception as error:
        write_json(attempt/"run_status.json",{"state":"failed","exception_type":type(error).__name__,
            "frozen_predictions_saved":(attempt/"frozen_plan_and_predictions.json").exists(),
            "final_result_saved":(attempt/"final_result.json").exists()})
        if not (attempt/"run_log.txt").exists():
            (attempt/"run_log.txt").write_text("Attempt failed: "+type(error).__name__+"\nUse run_status.json and this attempt only; earlier attempts are separate.\n",encoding="utf-8")
        raise
    write_json(attempt/"run_status.json",{"state":"complete"})
    return result


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
