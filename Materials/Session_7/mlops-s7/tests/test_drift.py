from conftest import make_fake_creditcard

from fraud.data import apply_concept_drift, apply_data_drift, load_raw

PARAMS = {"psi_alert": 0.1, "psi_retrain": 0.25, "pvalue": 0.01}


def _load(tmp_path, seed):
    path = tmp_path / f"cc-{seed}.csv"
    make_fake_creditcard(5000, seed=seed).to_csv(path, index=False)
    return load_raw(path)


def test_no_drift_on_same_distribution(tmp_path):
    from fraud.drift import compare

    result = compare(_load(tmp_path, 1), _load(tmp_path, 2), PARAMS)
    assert result["max_psi"] < 0.1 and not result["retrain"]


def test_drift_detected_on_shifted_inputs(tmp_path):
    from fraud.drift import compare

    result = compare(_load(tmp_path, 1), apply_data_drift(_load(tmp_path, 2), 0.8), PARAMS)
    assert result["retrain"]
    assert result["features"]["amount"]["drift_detected"]
    assert result["features"]["amount_band"]["drift_detected"]


def test_concept_drift_is_invisible_to_input_tests(tmp_path):
    """Disguised frauds do not change the inputs: distribution tests cannot see it."""
    from fraud.drift import compare

    result = compare(_load(tmp_path, 1), apply_concept_drift(_load(tmp_path, 2), 1.0), PARAMS)
    assert not result["retrain"]
