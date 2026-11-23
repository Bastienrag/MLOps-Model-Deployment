import yaml

from imu.repro import typed
from imu.track import flat, run_metrics
from imu.tracking import data_fingerprints, dvc_md5


def write_dvc(path, md5):
    path.write_text(yaml.safe_dump({"outs": [{"md5": md5, "path": path.stem}]}))


def test_dvc_md5_reads_the_pointer_file(tmp_path):
    f = tmp_path / "data.parquet.dvc"
    write_dvc(f, "abc123")
    assert dvc_md5(f) == "abc123"


def test_data_fingerprints_from_dvc_files_and_lock(tmp_path):
    write_dvc(tmp_path / "raw.parquet.dvc", "raw-md5")
    write_dvc(tmp_path / "labels.csv.dvc", "labels-md5")
    lock = {"stages": {"prepare": {"outs": [{"path": "data/processed", "md5": "proc-md5.dir"}]}}}
    (tmp_path / "dvc.lock").write_text(yaml.safe_dump(lock))
    params = {
        "raw_path": str(tmp_path / "raw.parquet"),
        "labels_path": str(tmp_path / "labels.csv"),
        "processed_dir": "data/processed",
    }
    assert data_fingerprints(params, tmp_path / "dvc.lock") == {
        "data_raw_md5": "raw-md5",
        "data_labels_md5": "labels-md5",
        "data_processed_md5": "proc-md5.dir",
    }


def test_typed_recovers_numbers_lists_and_strings():
    assert typed("0.001") == 0.001
    assert typed("[8, 16, 16]") == [8, 16, 16]
    assert typed("['Safe', 'Safe overall']") == ["Safe", "Safe overall"]
    assert typed("data/processed") == "data/processed"


def test_flat_prefixes_keys():
    assert flat("train", {"seed": 42}) == {"train.seed": 42}


def test_run_metrics_names_and_classes():
    m = {"f1_macro": 0.5, "balanced_accuracy": 0.4, "recall_per_class": {"Safe overall": 0.7}}
    results = {s: {"ride": m, "window": m} for s in ("val", "test")}
    out = run_metrics(results)
    assert out["test_ride_f1_macro"] == 0.5
    assert out["val_window_balanced_accuracy"] == 0.4
    assert out["test_ride_recall_safe_overall"] == 0.7
