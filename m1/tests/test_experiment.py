import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import experiment as m1
from snapshot import FEATURES, KEY, build_snapshot, split_snapshot


DISTINCT = Path(__file__).resolve().parents[1] / "samples" / "distinct_5k"


@pytest.fixture(scope="module")
def snapshot():
    return build_snapshot(DISTINCT)


def test_p1_fit_uses_only_train_and_binary_passthrough(snapshot):
    train, reference, keys, stats = m1.prepare(snapshot, 101)
    partitions = {row["cliente_id"]: row["split"] for row in split_snapshot(snapshot, seed=101)}
    train_values = [float(row["valor_mensal_total"]) for row in snapshot
                    if partitions[row["cliente_id"]] == "train"]
    assert stats["imputer_medians"]["valor_mensal_total"] == np.median(train_values)
    assert stats["scaler_scale"]["valor_mensal_total"] == np.subtract(*np.percentile(train_values, [75, 25]))
    expected = {tuple(row[field] for field in KEY): row[m1.BINARY] for row in snapshot}
    np.testing.assert_array_equal(reference[:, -1], [expected[key] for key in keys])

    modified = [{**row, "valor_mensal_total": 10**9}
                if partitions[row["cliente_id"]] != "train" else row for row in snapshot]
    for row in modified:
        if partitions[row["cliente_id"]] == "test":
            row["valor_mensal_total"] = "must not be evaluated"
    same_train, changed_reference, same_keys, same_stats = m1.prepare(modified, 101)
    np.testing.assert_array_equal(same_train, train)
    assert not np.array_equal(reference, changed_reference)
    assert same_keys == keys and same_stats == stats
    assert all(partitions[key[1]] == "reference" for key in keys)


@pytest.mark.parametrize("extra", ["renovou_90d", "perfil_latente", "another_id"])
def test_reject_unauthorized_feature(snapshot, extra):
    rows = [{**snapshot[0], extra: 0}, *snapshot[1:]]
    with pytest.raises(ValueError, match="unexpected or missing columns"):
        m1.prepare(rows, 101)
    assert not set(KEY).intersection(FEATURES)


def test_actual_models_are_repeatable(snapshot):
    train, ref, keys, _ = m1.prepare(snapshot, 101)
    with threadpool_limits(limits=2):
        for family in ("kmeans", "gmm"):
            first, labels, probabilities = m1.evaluate_fit(family, 2, 11, train, ref)
            second, repeat, _ = m1.evaluate_fit(family, 2, 11, train, ref)
            assert first["status"] == second["status"] != "failed"
            assert first["reasons"] == second["reasons"]
            if first["silhouette"] is None:
                assert second["silhouette"] is None
            else:
                assert first["silhouette"] == pytest.approx(second["silhouette"], abs=1e-12)
            np.testing.assert_array_equal(labels, repeat)
            assert sum(cluster["reference_count"] for cluster in first["clusters"]) == len(keys)
            assert (probabilities is None) == (family == "kmeans")
            if probabilities is not None:
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def fake_fit(family, k, seed, train, reference):
    labels = np.array([0, 0, 1, 1])
    return {"family": family, "k": k, "seed": seed, "status": "eligible",
            "silhouette": 0.40 + 0.002 * k, "min_reference_cluster": 30}, labels, None


def run_fake(tmp_path, *, budget=600, stop=lambda: False):
    keys = [("dataset", str(index), "date") for index in range(4)]
    return m1.run_search(None, None, keys, tmp_path, {}, budget_seconds=budget, stop_requested=stop)


def test_search_budget_and_ranking(monkeypatch, tmp_path):
    monkeypatch.setattr(m1, "evaluate_fit", fake_fit)
    report = run_fake(tmp_path)
    assert len(report["candidates"]) == 14
    assert len(report["finalists"]) == 3
    assert report["attempted_fits"] == 20
    assert report["selected"]["k"] == 2
    assert len(report["selected"]["pairwise_ari"]) == 3
    assert all(pair["ari"] == 1 for pair in report["selected"]["pairwise_ari"])
    counts = Counter(item["status"] for item in report["candidates"])
    assert counts == {"eligible": 14}
    with (tmp_path / "kmeans-2-seed-11" / "assignments.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 4


def test_stop_preserves_completed_fit(monkeypatch, tmp_path):
    stopped = False

    def fit(*args):
        nonlocal stopped
        stopped = True
        return fake_fit(*args)

    monkeypatch.setattr(m1, "evaluate_fit", fit)
    report = run_fake(tmp_path, stop=lambda: stopped)
    assert report["halt_reason"] == "signal"
    assert report["attempted_fits"] == 1
    assert report["selected"] is None
    assert (tmp_path / "kmeans-2-seed-11" / "assignments.csv").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["completed_fits"] == ["kmeans-2-seed-11"]


def test_timeout_preserves_completed_fit(monkeypatch, tmp_path):
    ticks = iter([0, 0, 601, 601])
    monkeypatch.setattr(m1, "evaluate_fit", fake_fit)
    monkeypatch.setattr(m1.time, "monotonic", lambda: next(ticks))
    report = run_fake(tmp_path)
    assert report["attempted_fits"] == 1
    assert report["halt_reason"] == "time_budget"
    assert (tmp_path / "kmeans-2-seed-11" / "metrics.json").exists()


def test_no_eligible_candidates(monkeypatch, tmp_path):
    def rejected(*args):
        result, _, _ = fake_fit(*args)
        return {**result, "status": "rejected", "reasons": ["degenerate_cluster"]}, None, None

    monkeypatch.setattr(m1, "evaluate_fit", rejected)
    report = run_fake(tmp_path)
    assert report["attempted_fits"] == 14
    assert report["selected"] is None and report["finalists"] == []
