"""M2.2A: five prespecified clustering controls on distinct_5k."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import signal
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from threadpoolctl import threadpool_limits

from experiment import (BINARY, MODEL_PARAMS, evaluate_fit, feature_array, fit_p1,
                        partition_rows, persist_fit, write_json)
from sensitivity import (INIT_SEEDS, QUANTILES, SCREEN_SEED, SPLIT_SEED, SUBSAMPLE_SEEDS,
                         characterize, digest, prepare_representation, rss_bytes, subsample,
                         summarize_stability)
from snapshot import FEATURES, KEY, build_snapshot, load_manifest

ROOT = Path(__file__).resolve().parent
HISTORICAL = ROOT.parent / "m2.1" / "results" / "distinct"
USAGE = "qtd_eventos_uso"
VALUE = "valor_mensal_total"
CANDIDATES = {
    "C1": ("kmeans", "P2", ()),
    "C2": ("kmeans", "P2", (BINARY,)),
    "C3": ("kmeans", "P2", (BINARY, USAGE)),
    "C4": ("kmeans", "P1_upper_usage", ()),
    "C5": ("gmm", "P2", ()),
}
MAX_FITS = 55
MAX_SECONDS = 600


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clip_observations(train, reference):
    """Cap usage above the training quantile. Keep zeros and NaNs before scaling."""
    matrices = [train.copy(), reference.copy()]
    limits = {}
    counts = {}
    affected = [np.zeros(len(matrix), dtype=bool) for matrix in matrices]
    for field in (VALUE, USAGE):
        column = FEATURES.index(field)
        observed = train[:, column]
        if np.isnan(observed).all():
            raise ValueError(f"No observed values for {field}")
        lower, upper = np.nanquantile(observed, QUANTILES).tolist()
        if field == USAGE:
            lower = None
        limits[field] = [lower, upper]
        counts[field] = []
        for matrix, any_change in zip(matrices, affected, strict=True):
            values = matrix[:, column]
            changed = values > upper
            if lower is not None:
                changed |= values < lower
            counts[field].append(int(changed.sum()))
            any_change |= changed
            matrix[:, column] = np.clip(values, lower, upper)
    return (*matrices, limits, counts, affected)


def prepare(candidate, train, reference):
    _, representation, dropped = CANDIDATES[candidate]
    if representation == "P1_upper_usage":
        capped_train, capped_ref, limits, counts, affected = clip_observations(train, reference)
        tx, rx, stats = fit_p1(capped_train, capped_ref)
        stats.update(limits=limits, affected_train_reference_by_field=counts,
                     affected_train_reference=[int(mask.sum()) for mask in affected])
    else:
        tx, rx, stats = prepare_representation(representation, train, reference)
    retained = [i for i, field in enumerate(stats["features_out"]) if field not in dropped]
    stats["model_features"] = [stats["features_out"][i] for i in retained]
    stats["dropped"] = list(dropped)
    # P2's imputation and scaling are columnwise. Dropping transformed columns
    # leaves the retained values identical to fitting those columns alone.
    return tx[:, retained], rx[:, retained], stats


def compare(left, right):
    if len(left) != len(right):
        raise ValueError("partitions must cover the same reference")
    return {
        "ari": float(adjusted_rand_score(left, right)),
        "left_labels": np.unique(left).tolist(), "right_labels": np.unique(right).tolist(),
        "contingency": contingency_matrix(left, right).tolist(),
    }


def load_historical(directory, keys, source_sha):
    manifest = json.loads((directory / "manifest.json").read_text())
    artifact = "P1_clip-kmeans-2-seed-11"
    if (manifest["source_manifest_sha256"] != source_sha
            or manifest["reference_keys_sha256"] != digest(keys)
            or artifact not in manifest["completed_fits"]):
        raise ValueError("historical fit does not match source and reference")
    path = directory / artifact / "assignments.csv"
    labels_by_key = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[field] for field in KEY)
            label = int(row["cluster"])
            if key in labels_by_key or label not in (0, 1):
                raise ValueError("invalid historical assignment")
            labels_by_key[key] = label
    if labels_by_key.keys() != set(keys) or len(keys) != len(labels_by_key):
        raise ValueError("historical keys differ from reference")
    return np.array([labels_by_key[key] for key in keys]), {
        "artifact": str(path), "assignments_sha256": sha256(path),
        "manifest_sha256": sha256(directory / "manifest.json"),
    }


def baseline_partitions(train, reference):
    cutoff = float(np.median(train[:, FEATURES.index(VALUE)]))
    value = (reference[:, FEATURES.index(VALUE)] >= cutoff).astype(int)
    coverage = reference[:, FEATURES.index(BINARY)].astype(int)
    return {"value": value, "coverage": coverage, "value_coverage": 2 * value + coverage}, cutoff


def stability_report(initial, runs):
    all_runs = summarize_stability(initial, runs, 2)
    eligible = [(record, labels) for record, labels in runs if record["status"] == "eligible"]
    diagnostics = summarize_stability(initial, eligible, 2)
    # Only the complete population of runs defines the inherited stability gate.
    eligible_only = {field: diagnostics[field] for field in (
        "runs", "ari_all", "ari_between_subsamples", "pairs", "aligned_group_shares")}
    return {"all_runs": all_runs, "eligible_only": eligible_only,
            "runs": [{"artifact": record["artifact"], "status": record["status"],
                      "reasons": record.get("reasons", []), "clusters": record.get("clusters", []),
                      "vanished_groups": record.get("vanished_groups", [])}
                     for record, _ in runs]}


def run(train, reference, train_keys, ref_keys, historical, out, manifest, *, budget, stopped):
    started = time.monotonic()
    completed = []
    initial = {}
    resamples = {code: [] for code in CANDIDATES}
    halt = None

    def can_start():
        nonlocal halt
        if stopped():
            halt = "signal"
        elif time.monotonic() - started >= budget:
            halt = "time_budget"
        elif len(completed) >= MAX_FITS:
            halt = "fit_budget"
        return halt is None

    def fit(code, seed, indices=None, sample_seed=None):
        raw_train = train if indices is None else train[indices]
        tx, rx, stats = prepare(code, raw_train, reference)
        family, representation, _ = CANDIDATES[code]
        record, labels, probabilities = evaluate_fit(family, 2, seed, tx, rx)
        artifact = f"{code}-seed-{seed}" + (f"-sample-{sample_seed}" if indices is not None else "")
        record.update(candidate=code, representation=representation, artifact=artifact,
                      subsample_seed=sample_seed, preprocessing=stats,
                      train_rows=len(tx), reference_rows=len(rx), peak_process_rss_bytes=rss_bytes(),
                      silhouette_space="Euclidean on candidate's fitted features, fixed reference")
        if labels is not None:
            record["vanished_groups"] = [
                group["cluster"] for group in record["clusters"]
                if group["train_count"] and not group["reference_count"]
            ]
            if indices is None:
                record["reference_profiles"] = characterize(
                    reference, labels, 2, np.zeros(len(reference), dtype=bool))
                for profile in record["reference_profiles"]:
                    del profile["outside_train_cutoffs"]
        persist_fit(out / artifact, record, ref_keys, labels, probabilities)
        completed.append(artifact)
        manifest["completed_fits"] = completed
        write_json(out / "manifest.json", manifest)
        return record, labels

    samples = {seed: subsample(len(train), seed) for seed in SUBSAMPLE_SEEDS}
    manifest["subsamples"] = {
        seed: {"rows": len(indices), "keys_sha256": digest([train_keys[i] for i in indices])}
        for seed, indices in samples.items()
    }
    write_json(out / "manifest.json", manifest)
    for code in CANDIDATES:
        if not can_start():
            break
        initial[code] = fit(code, SCREEN_SEED)

    for code in initial if halt is None else ():
        for sample_seed, seed in itertools.product(SUBSAMPLE_SEEDS, INIT_SEEDS):
            if not can_start():
                break
            resamples[code].append(fit(code, seed, samples[sample_seed], sample_seed))
        if halt:
            break

    if stopped():
        halt = "signal"
    elif halt is None and time.monotonic() - started >= budget:
        halt = "time_budget"
    baseline, cutoff = baseline_partitions(train, reference)
    partitions = {code: labels for code, (_, labels) in initial.items() if labels is not None}
    partitions["P1_clip_historical"] = historical
    comparisons = {
        f"{left} x {right}": compare(partitions[left], partitions[right])
        for left, right in (("C1", "C2"), ("C1", "C3"), ("C4", "P1_clip_historical"),
                            ("C1", "C4"), ("C1", "C5"))
        if left in partitions and right in partitions
    }
    stability = {
        code: stability_report(labels, resamples[code])
        for code, labels in partitions.items() if code in CANDIDATES
    }
    report = {
        "initial": {code: record for code, (record, _) in initial.items()},
        "stability": stability,
        "resample_attempts": {code: [{"artifact": record["artifact"], "status": record["status"],
                                    "reasons": record["reasons"]}
                                   for record, _ in runs] for code, runs in resamples.items()},
        "comparisons": comparisons,
        "baselines": {
            "value_cutoff_train": cutoff,
            "groups": {name: {"counts": np.bincount(labels, minlength=4 if name == "value_coverage" else 2).tolist(),
                              "profiles": [{k: v for k, v in profile.items() if k != "outside_train_cutoffs"}
                                           for profile in characterize(reference, labels,
                                               4 if name == "value_coverage" else 2,
                                               np.zeros(len(reference), dtype=bool))]}
                       for name, labels in baseline.items()},
            "comparisons": {code: {name: compare(labels, groups) for name, groups in baseline.items()}
                            for code, labels in partitions.items() if code in CANDIDATES},
        },
        "attempted_fits": len(completed), "halt_reason": halt,
        "passes_stability": [code for code, metrics in stability.items()
                             if metrics["all_runs"]["passes_stability_gate"]],
        "search_seconds": time.monotonic() - started, "peak_process_rss_bytes": rss_bytes(),
    }
    write_json(out / "report.json", report)
    manifest.update(status="stopped" if halt else "complete", halt_reason=halt)
    write_json(out / "manifest.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--budget-seconds", type=float, default=MAX_SECONDS)
    args = parser.parse_args()
    if not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= MAX_SECONDS:
        parser.error("budget must be finite and in (0, 600]")
    started = time.perf_counter()
    stopping = False

    def stop(_number, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    path = ROOT / "samples" / "distinct_5k"
    source, dataset_id = load_manifest(path)
    train_rows, ref_rows = partition_rows(build_snapshot(path, expected_dataset_id=dataset_id), SPLIT_SEED)
    train, reference = feature_array(train_rows), feature_array(ref_rows)
    train_keys, ref_keys = [[tuple(row[field] for field in KEY) for row in rows]
                            for rows in (train_rows, ref_rows)]
    source_sha = sha256(path / "manifest.json")
    historical, provenance = load_historical(HISTORICAL, ref_keys, source_sha)
    manifest = {
        "status": "running", "dataset_id": dataset_id, "source": source,
        "source_manifest_sha256": source_sha, "historical": provenance,
        "split_seed": SPLIT_SEED, "train_keys_sha256": digest(train_keys),
        "reference_keys_sha256": digest(ref_keys),
        "population": {"train": len(train), "reference": len(reference),
                       "reserved": source["n_clients"] - len(train) - len(reference)},
        "screen_seed": SCREEN_SEED, "init_seeds": INIT_SEEDS, "subsample_seeds": SUBSAMPLE_SEEDS,
        "subsample_fraction": .8, "with_replacement": False,
        "candidates": CANDIDATES, "k": 2, "model_parameters": MODEL_PARAMS,
        "stability_gate": {"min_eligible": 9, "ari_median": .75, "ari_p10": .60,
                           "max_share_spread_pp": 10, "allow_disappearance": False},
        "budget": {"fits": MAX_FITS, "seconds": args.budget_seconds}, "threads": 2,
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {line.split("==")[0]: version(line.split("==")[0])
                         for line in (ROOT / "requirements.txt").read_text().splitlines()},
        "code_sha256": {name: sha256(ROOT / name) for name in
                        ("controls.py", "sensitivity.py", "experiment.py", "snapshot.py")},
        "completed_fits": [],
    }
    args.out.mkdir(parents=True)
    with threadpool_limits(limits=2):
        report = run(train, reference, train_keys, ref_keys, historical, args.out, manifest,
                     budget=args.budget_seconds, stopped=lambda: stopping)
    report["total_seconds"] = time.perf_counter() - started
    write_json(args.out / "report.json", report)
    print(f"{report['attempted_fits']} fits; status={manifest['status']}")


if __name__ == "__main__":
    main()
