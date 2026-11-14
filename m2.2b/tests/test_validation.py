import json

import numpy as np
import pytest

import validation as v


def test_rate_and_bootstrap():
    labels = np.array([0, 0, 1, 1])
    events = np.array([0, 1, 1, 1])
    result = v.group_summary(labels, events)
    assert result == v.group_summary(labels, events)
    assert result["groups"][0]["events"] == 1
    assert result["groups"][0]["difference_pp"] == -50
    assert result["groups"][1]["risk_ratio"] == 2
    assert result["groups"][0]["ci95_pp"][0] <= -50 <= result["groups"][0]["ci95_pp"][1]


def test_rules_distinguish_zero_missing_and_use_frozen_limits():
    raw = np.zeros((5, len(v.FEATURES)))
    raw[:, v.FEATURES.index(v.USAGE)] = [0, np.nan, 7, 18, 17]
    raw[:, v.FEATURES.index(v.BINARY)] = [1, 0, 1, 1, 1]
    raw[:, v.FEATURES.index(v.VALUE)] = [0, 30, 1, 20, 10]
    result = v.rules(raw, 10)
    assert result["low_use"].tolist() == [1, 0, 1, 0, 0]
    assert result["high_use"].tolist() == [0, 0, 0, 1, 0]
    assert result["value"].tolist() == [0, 1, 0, 1, 1]
    assert result["zero"].tolist() == [1, 0, 0, 0, 0]


def test_splits_isolated_and_reproduced():
    all_ids = set()
    for scenario in v.SCENARIOS:
        _, _, pops = v.population(scenario, "train", "reference", "test")
        assert {part: len(rows) for part, rows in pops.items()} == {"train": 3000, "reference": 1000, "test": 1000}
        keys = [v.key(row) for rows in pops.values() for row in rows]
        assert len(set(keys)) == 5000 and not set(keys) & all_ids
        all_ids.update(keys)
        assert pops == v.population(scenario, "train", "reference", "test")[2]


def test_outcome_join_contract_and_scope(monkeypatch):
    source = v.ROOT / "samples" / "distinct_5k"
    _, dataset = v.load_manifest(source)
    row = dict(zip(v.KEY, (dataset, "S000000001", "2026-09-30")))
    data = {"cliente_id": row["cliente_id"], "janela_fim": "2026-12-31",
            **dict.fromkeys(v.OUTCOMES, "1")}
    monkeypatch.setattr(v, "read_table", lambda *_: iter([{"cliente_id": "another"}, data]))
    assert v.outcomes(source, [row]).tolist() == [[1, 1, 1]]
    for change in ({"dataset_id": "wrong"}, {"data_corte": "2026-10-01"}):
        with pytest.raises(ValueError, match="dataset"):
            v.outcomes(source, [{**row, **change}])
    data["janela_fim"] = "2026-12-30"
    with pytest.raises(ValueError, match="window"):
        v.outcomes(source, [row])


def test_fit_stops_and_preserves_artifacts(monkeypatch, tmp_path):
    calls = 0

    def fit(_family, _k, _seed, tx, rx, **_kwargs):
        nonlocal calls
        calls += 1
        return {"status": "failed"}, None, None

    monkeypatch.setattr(v, "evaluate_fit", fit)
    monkeypatch.setattr(v, "outcomes", lambda *_: pytest.fail("Training touched outcomes"))
    path = tmp_path / "run"
    result = v.train_all(path, 600, lambda: calls == 1)
    assert result["status"] == "stopped" and len(result["completed"]) == 1
    assert (path / result["completed"][0] / "metrics.json").exists()
    assert (path / "preregistration.json").exists()
    assert not (path / "test_opened.json").exists()


def test_single_open_and_calibration_overlap(tmp_path):
    path = tmp_path / "marker.json"
    v.claim(path, {"opening": 1})
    with pytest.raises(FileExistsError):
        v.claim(path, {"opening": 2})
    assert json.loads(path.read_text()) == {"opening": 1}
    row = dict(zip(v.KEY, ("dataset", "client", "cutoff")))
    with pytest.raises(ValueError, match="overlap"):
        v.evaluate_population(None, [row], {}, None, calibration={"keys": [v.key(row)]})


def test_brier_and_selection_are_reference_based():
    n = 100
    y = np.arange(n) % 2
    groups = {rule: y for rule in v.RULES}
    scores = v.brier(groups, y, "zero", np.arange(n) % 5)
    assert scores["passes"] and scores["adjusted_ci"][0] > 0
    assert scores == v.brier(groups, y, "zero", np.arange(n) % 5)
    fitted = {"scenario": {s: {"models": {m: {"status": "failed"} for m in v.MODELS}}
                            for s in v.SCENARIOS}}
    reference = {s: {target: {"methods": {r: {"brier": {
        "passes": r == "zero" and target == v.OUTCOMES[0], "gain": .01}}
        for r in v.RULES}} for target in v.OUTCOMES} for s in v.SCENARIOS}
    result = v.select(reference, fitted)
    assert result["method"] == "zero" and result["outcome"] == v.OUTCOMES[0]
    reference["continuum"] = {}
    assert result == v.select(reference, fitted)
