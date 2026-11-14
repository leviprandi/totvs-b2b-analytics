import csv
import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import controls as m2
from experiment import feature_array, partition_rows
from sensitivity import digest, prepare_representation
from snapshot import FEATURES, KEY, build_snapshot, split_snapshot


DATASET = m2.ROOT / "samples" / "distinct_5k"


@pytest.fixture(scope="module")
def partitions():
    rows = build_snapshot(DATASET)
    train, reference = partition_rows(rows, m2.SPLIT_SEED)
    keys = [[tuple(row[field] for field in KEY) for row in part] for part in (train, reference)]
    return rows, feature_array(train), feature_array(reference), *keys


def test_upper_cap_preserves_zero_positive_and_absence(partitions):
    _, train, reference, *_ = partitions
    ref = reference[:4].copy()
    usage = FEATURES.index(m2.USAGE)
    ref[:, usage] = [0, 3, 1e9, np.nan]
    ref[:, FEATURES.index(m2.BINARY)] = [1, 1, 1, 0]
    _, capped, limits, counts, _ = m2.clip_observations(train, ref)
    assert limits[m2.USAGE][0] is None
    np.testing.assert_array_equal(capped[:, usage], [0, 3, limits[m2.USAGE][1], np.nan])
    assert counts[m2.USAGE][1] == 1
    _, scaled, stats = m2.prepare("C4", train, ref)
    np.testing.assert_array_equal(scaled[:, -1], [1, 1, 1, 0])
    col = stats["model_features"].index(m2.USAGE)
    center, scale = stats["scaler_center"][m2.USAGE], stats["scaler_scale"][m2.USAGE]
    assert scaled[0, col] * scale + center == 0


@pytest.mark.parametrize("code", m2.CANDIDATES)
def test_transforms_and_ablations_are_train_only(partitions, code):
    _, train, reference, *_ = partitions
    tx, rx, stats = m2.prepare(code, train, reference)
    changed = reference.copy()
    changed[:, FEATURES.index(m2.VALUE)] = 1e9
    modified_tx, modified_rx, updated = m2.prepare(code, train, changed)
    np.testing.assert_array_equal(tx, modified_tx)
    assert not np.array_equal(rx, modified_rx)
    assert stats["scaler_center"] == updated["scaler_center"]
    assert stats["scaler_scale"] == updated["scaler_scale"]
    assert stats["imputer_medians"] == updated["imputer_medians"]
    assert stats["limits"] == updated["limits"]
    assert set(stats["model_features"]) == set(FEATURES) - set(m2.CANDIDATES[code][2])
    if code in ("C2", "C3"):
        full, _, original = m2.prepare("C1", train, reference)
        retained = [original["model_features"].index(field) for field in stats["model_features"]]
        np.testing.assert_array_equal(tx, full[:, retained])


def test_reserved_values_are_never_materialized_for_fit(partitions):
    rows, train, reference, *_ = partitions
    reserved = {row["cliente_id"] for row in split_snapshot(rows, seed=m2.SPLIT_SEED)
                if row["split"] == "test"}
    modified = [{**row, m2.VALUE: "unreadable"} if row["cliente_id"] in reserved else row for row in rows]
    actual_train, actual_ref = partition_rows(modified, m2.SPLIT_SEED)
    np.testing.assert_array_equal(feature_array(actual_train), train)
    np.testing.assert_array_equal(feature_array(actual_ref), reference)


def test_historical_regression(partitions):
    _, train, reference, _, keys = partitions
    before = m2.sha256(m2.HISTORICAL / "manifest.json")
    labels, provenance = m2.load_historical(m2.HISTORICAL, keys, m2.sha256(DATASET / "manifest.json"))
    tx, rx, _ = prepare_representation("P1_clip", train, reference)
    with threadpool_limits(limits=2):
        result, replay, _ = m2.evaluate_fit("kmeans", 2, 11, tx, rx)
    np.testing.assert_array_equal(replay, labels)
    old = json.loads(Path(provenance["artifact"]).with_name("metrics.json").read_text())
    assert result["silhouette"] == pytest.approx(old["silhouette"], abs=1e-12)
    assert before == m2.sha256(m2.HISTORICAL / "manifest.json")
    with pytest.raises(ValueError, match="does not match"):
        m2.load_historical(m2.HISTORICAL, keys[:-1], m2.sha256(DATASET / "manifest.json"))


@pytest.mark.parametrize("code", m2.CANDIDATES)
def test_real_subsample_and_fit_repeat(partitions, code):
    _, train, reference, train_keys, _ = partitions
    sample = m2.subsample(len(train), 1011)
    assert len(set(sample)) == 2400
    assert digest([train_keys[i] for i in sample]) == digest(
        [train_keys[i] for i in m2.subsample(len(train), 1011)])
    tx, rx, _ = m2.prepare(code, train[sample], reference)
    with threadpool_limits(limits=2):
        first, a, _ = m2.evaluate_fit(m2.CANDIDATES[code][0], 2, 11, tx, rx)
        second, b, _ = m2.evaluate_fit(m2.CANDIDATES[code][0], 2, 11, tx, rx)
    assert first["status"] == second["status"] != "failed"
    assert first["silhouette"] == pytest.approx(second["silhouette"])
    np.testing.assert_array_equal(a, b)


def test_ari_and_degenerate_runs():
    initial = np.array([0, 0, 1, 1])
    assert m2.compare(initial, 1 - initial)["ari"] == 1
    assert m2.compare(initial, 1 - initial)["contingency"] == [[0, 2], [2, 0]]
    runs = [({"artifact": str(i), "subsample_seed": i // 2, "status": "eligible"}, initial)
            for i in range(9)]
    runs.append(({"artifact": "9", "subsample_seed": 4, "status": "rejected",
                  "reasons": ["degenerate_cluster"]}, np.zeros(4, dtype=int)))
    summary = m2.stability_report(initial, runs)
    assert not summary["all_runs"]["passes_stability_gate"]
    assert summary["all_runs"]["ari_between_subsamples"]["pairs"] == 40
    assert summary["eligible_only"]["ari_between_subsamples"]["pairs"] == 32


@pytest.mark.parametrize("limited", [False, True])
def test_fit_budget_keys_and_preservation(partitions, monkeypatch, tmp_path, limited):
    _, train, reference, train_keys, ref_keys = partitions
    calls = 0

    def fit(family, k, seed, tx, rx):
        nonlocal calls
        calls += 1
        labels = np.arange(len(rx)) % 2
        return {"family": family, "k": k, "seed": seed, "status": "eligible", "reasons": [],
                "clusters": [{"cluster": i, "train_count": len(tx) // 2,
                              "reference_count": len(rx) // 2} for i in range(2)]}, labels, None

    monkeypatch.setattr(m2, "evaluate_fit", fit)
    if limited:
        monkeypatch.setattr(m2.time, "monotonic", lambda: 601 if calls else 0)
    result = m2.run(train, reference, train_keys, ref_keys, np.arange(len(reference)) % 2,
                    tmp_path, {}, budget=600, stopped=lambda: False)
    assert calls == result["attempted_fits"] == (1 if limited else 55)
    assert result["halt_reason"] == ("time_budget" if limited else None)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["completed_fits"]) == calls
    for artifact in manifest["completed_fits"]:
        with (tmp_path / artifact / "assignments.csv").open(newline="") as stream:
            assert [tuple(row[field] for field in KEY) for row in csv.DictReader(stream)] == ref_keys
