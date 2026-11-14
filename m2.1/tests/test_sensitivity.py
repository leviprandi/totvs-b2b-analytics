import json
from pathlib import Path

import numpy as np
import pytest

import sensitivity as m2
from experiment import BINARY, feature_array, fit_p1, partition_rows
from snapshot import FEATURES, build_snapshot, split_snapshot


@pytest.fixture(scope="module")
def observed():
    rows = build_snapshot(m2.ROOT / "samples" / "distinct_5k")
    train, reference = partition_rows(rows, 101)
    return rows, feature_array(train), feature_array(reference)


def test_p1_matches_m1_transform(observed):
    _, train, reference = observed
    actual = m2.prepare_representation("P1", train, reference)
    baseline = fit_p1(train, reference)
    np.testing.assert_array_equal(actual[0], baseline[0])
    np.testing.assert_array_equal(actual[1], baseline[1])
    assert all(actual[2][key] == value for key, value in baseline[2].items())


@pytest.mark.parametrize("representation", ["P2", "P1_clip"])
def test_transforms_learn_only_from_train(observed, representation):
    _, train, reference = observed
    _, _, original = m2.prepare_representation(representation, train, reference)
    reference = reference.copy()
    column = FEATURES.index("valor_mensal_total")
    reference[:, column] = 10**9
    _, transformed, updated = m2.prepare_representation(representation, train, reference)
    assert original == updated
    np.testing.assert_array_equal(transformed[:, -1], reference[:, FEATURES.index(BINARY)])
    if representation == "P1_clip":
        for field, bounds in original["limits"].items():
            np.testing.assert_allclose(bounds, np.nanquantile(train[:, FEATURES.index(field)], m2.QUANTILES))
        upper = original["limits"]["valor_mensal_total"][1]
        expected = (upper - original["scaler_center"]["valor_mensal_total"]) / original["scaler_scale"]["valor_mensal_total"]
        np.testing.assert_allclose(transformed[:, 0], expected)
    else:
        assert original["imputer_medians"]["valor_mensal_total"] == np.median(np.log1p(train[:, column]))
        assert original["imputer_medians"]["nps_medio"] == np.nanmedian(train[:, FEATURES.index("nps_medio")])
        assert original["imputer_medians"]["proporcao_gratuito"] == np.median(train[:, FEATURES.index("proporcao_gratuito")])
        reference[0, column] = -1
        with pytest.raises(ValueError, match="nonnegative"):
            m2.prepare_representation(representation, train, reference)


def test_subsamples_are_distinct_reproducible_and_exclude_reserved(observed):
    rows, _, _ = observed
    train, _ = partition_rows(rows, 101)
    partitions = {row["cliente_id"]: row["split"] for row in split_snapshot(rows, seed=101)}
    samples = []
    for seed in m2.SUBSAMPLE_SEEDS:
        indices = m2.subsample(len(train), seed)
        assert len(indices) == len(set(indices)) == 2400
        np.testing.assert_array_equal(indices, m2.subsample(len(train), seed))
        assert all(partitions[train[i]["cliente_id"]] == "train" for i in indices)
        samples.append(tuple(indices))
    assert len(set(samples)) == 5


def test_alignment_and_cross_subsample_ari():
    baseline = np.array([0, 0, 0, 1, 1, 1])
    runs = [({"artifact": f"{i}", "subsample_seed": i // 2, "status": "eligible"},
             baseline if i % 2 else 1 - baseline) for i in range(10)]
    result = m2.summarize_stability(baseline, runs, 2)
    assert result["ari_all"]["pairs"] == 45
    assert result["ari_between_subsamples"] == {"pairs": 40, "median": 1, "p10": 1, "std": 0}
    assert result["passes_stability_gate"]
    runs[-1] = ({**runs[-1][0], "status": "rejected"}, np.zeros(6, dtype=int))
    result = m2.summarize_stability(baseline, runs, 2)
    assert "cluster_share_variation_or_disappearance" in result["reasons"]


def test_search_budget_and_checkpoint(monkeypatch, tmp_path, observed):
    rows, train, reference = observed
    train_rows, ref_rows = partition_rows(rows, 101)
    key_lists = [[(row["dataset_id"], row["cliente_id"], row["data_corte"]) for row in partition]
                 for partition in (train_rows, ref_rows)]
    count = 0

    def fake_fit(family, k, seed, tx, rx, *, include_train_labels):
        nonlocal count
        count += 1
        labels = np.arange(len(rx)) % k
        train_labels = np.arange(len(tx)) % k
        result = {"status": "eligible", "family": family, "k": k, "seed": seed,
                  "silhouette": .5, "min_reference_cluster": int(np.bincount(labels).min()),
                  "clusters": [{"cluster": group, "train_count": int(sum(train_labels == group)),
                                "reference_count": int(sum(labels == group))} for group in range(k)]}
        if include_train_labels:
            result["train_labels"] = train_labels.tolist()
        return result, labels, None

    monkeypatch.setattr(m2, "evaluate_fit", fake_fit)
    out = tmp_path / "full"
    out.mkdir()
    full = m2.run(train, reference, *key_lists, out, {}, budget=600, stopped=lambda: False)
    assert count == full["screen_fits"] + full["resample_fits"] == 82
    assert len(full["screen"]) == 42 and len(full["finalists"]) == 4
    assert {item["representation"] for item in full["finalists"]} == set(m2.REPRESENTATIONS)
    assert {item["family"] for item in full["finalists"]} == set(m2.MODEL_PARAMS)
    out = tmp_path / "stopped"
    out.mkdir()
    partial = m2.run(train, reference, *key_lists, out, {}, budget=600, stopped=lambda: count >= 83)
    assert partial["screen_fits"] == 1 and partial["resample_fits"] == 0
    assert partial["halt_reason"] == "signal" and partial["robust_candidates"] == []
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["completed_fits"] == ["P1-kmeans-2-seed-11"]
    assert (out / manifest["completed_fits"][0] / "assignments.csv").exists()
