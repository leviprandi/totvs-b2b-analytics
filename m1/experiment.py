"""M1: bounded K-Means and diagonal GMM search on the M0 distinct snapshot."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import resource
import signal
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

from snapshot import FEATURES, KEY, build_snapshot, load_manifest, split_snapshot, training_matrix


BINARY = "telemetria_observada"
NUMERIC = tuple(field for field in FEATURES if field != BINARY)
SEEDS = (11, 29, 47)
CONFIGS = tuple((family, k) for family in ("kmeans", "gmm") for k in range(2, 9))
MODEL_PARAMS = {
    "kmeans": {"init": "k-means++", "n_init": 10, "max_iter": 300, "tol": 1e-4},
    "gmm": {"covariance_type": "diag", "n_init": 5, "max_iter": 300, "tol": 1e-3,
            "reg_covar": 1e-5, "init_params": "kmeans"},
}
MAX_FITS = 20
SILHOUETTE_BAND = 0.02
MIN_ARI = 0.75


def feature_array(rows):
    matrix = np.asarray([[row[field] for field in FEATURES] for row in training_matrix(rows)],
                        dtype=np.float64)
    binary = matrix[:, FEATURES.index(BINARY)]
    if np.isinf(matrix).any() or not np.isin(binary, (0, 1)).all():
        raise ValueError("P1 requires finite values or numeric NaN and binary coverage 0/1")
    usage = matrix[:, FEATURES.index("qtd_eventos_uso")]
    if not np.array_equal(np.isnan(usage), binary == 0):
        raise ValueError("P1 usage must be missing exactly when telemetry is absent")
    return matrix


def prepare(rows, split_seed):
    rows = sorted(rows, key=lambda row: tuple(row[field] for field in KEY))
    partitions = split_snapshot(rows, seed=split_seed)
    train_rows, ref_rows = [], []
    for row, partition in zip(rows, partitions, strict=True):
        if partition["split"] == "train":
            train_rows.append(row)
        elif partition["split"] == "reference":
            ref_rows.append(row)
    train, reference = feature_array(train_rows), feature_array(ref_rows)
    if np.isnan(train).all(axis=0).any():
        raise ValueError("P1 cannot estimate a median for an all-missing training feature")
    numeric_indices = [FEATURES.index(field) for field in NUMERIC]
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", RobustScaler())]), numeric_indices),
        ("binary", "passthrough", [FEATURES.index(BINARY)]),
    ])
    transformed_train = transformer.fit_transform(train)
    transformed_ref = transformer.transform(reference)
    pipeline = transformer.named_transformers_["numeric"]
    stats = {
        "features_in": list(FEATURES), "features_out": [*NUMERIC, BINARY],
        "imputer_medians": dict(zip(NUMERIC, pipeline["imputer"].statistics_.tolist())),
        "scaler_center": dict(zip(NUMERIC, pipeline["scaler"].center_.tolist())),
        "scaler_scale": dict(zip(NUMERIC, pipeline["scaler"].scale_.tolist())),
        "binary_passthrough": BINARY,
    }
    keys = [tuple(row[field] for field in KEY) for row in ref_rows]
    return transformed_train, transformed_ref, keys, stats


def evaluate_fit(family, k, seed, train, reference):
    started = time.perf_counter()
    params = {**MODEL_PARAMS[family], "random_state": seed}
    component_param = "n_clusters" if family == "kmeans" else "n_components"
    params[component_param] = k
    result = {"family": family, "k": k, "seed": seed, "parameters": params}
    labels = probabilities = None
    captured = []
    fit_started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = (KMeans if family == "kmeans" else GaussianMixture)(**params)
            model.fit(train)
        result["fit_seconds"] = time.perf_counter() - fit_started
        labels = model.predict(reference)
        train_labels = model.labels_ if family == "kmeans" else model.predict(train)
        counts = np.bincount(labels, minlength=k)
        train_counts = np.bincount(train_labels, minlength=k)
        result["iterations"] = int(model.n_iter_)
        minimum = max(30, math.ceil(0.03 * len(reference)))
        reasons = []
        converged = (model.n_iter_ < params["max_iter"] if family == "kmeans"
                     else bool(model.converged_))
        if not converged or any(issubclass(item.category, ConvergenceWarning) for item in captured):
            reasons.append("convergence_not_confirmed")
        if counts.min() < minimum or train_counts.min() == 0:
            reasons.append("degenerate_cluster")
        scores = silhouette_samples(reference, labels) if 1 < np.count_nonzero(counts) < len(labels) else None
        silhouette = float(scores.mean()) if scores is not None else None
        if silhouette is None or silhouette <= 0:
            reasons.append("nonpositive_or_undefined_silhouette")
        result.update(
            converged=converged, silhouette=silhouette, min_reference_cluster=int(counts.min()),
            clusters=[
                {"cluster": i, "train_count": int(train_counts[i]), "reference_count": int(counts[i]),
                 "silhouette": float(scores[labels == i].mean()) if scores is not None and counts[i] else None}
                for i in range(k)
            ],
        )
        if family == "kmeans":
            result["inertia_train"] = float(model.inertia_)
        else:
            probabilities = model.predict_proba(reference)
            if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
                raise ValueError("invalid GMM membership probabilities")
            result.update(aic_train=float(model.aic(train)), bic_train=float(model.bic(train)),
                          mean_max_membership_reference=float(probabilities.max(axis=1).mean()))
        result.update(status="rejected" if reasons else "eligible", reasons=reasons)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        result.update(status="failed", reasons=["fit_or_evaluation_failed"],
                      error=f"{type(exc).__name__}: {exc}")
        result.setdefault("fit_seconds", time.perf_counter() - fit_started)
        labels = probabilities = None
    result["warnings"] = [f"{item.category.__name__}: {item.message}" for item in captured]
    result["elapsed_seconds"] = time.perf_counter() - started
    return result, labels, probabilities


def candidate_order(candidates):
    """Group near-ties against the best remaining silhouette before ordering by simplicity."""
    remaining = list(candidates)
    ordered = []
    while remaining:
        best = max(item["selection_silhouette"] for item in remaining)
        band = [item for item in remaining if item["selection_silhouette"] >= best - SILHOUETTE_BAND]
        ordered.extend(sorted(band, key=lambda item: (
            item["k"], -item["min_reference_cluster"], -item["selection_silhouette"], item["family"],
        )))
        remaining = [item for item in remaining if item not in band]
    return ordered


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def persist_fit(directory, result, keys, labels, probabilities):
    pending = directory.with_name(directory.name + ".pending")
    pending.mkdir()
    write_json(pending / "metrics.json", result)
    if labels is not None:
        prob_columns = ([f"prob_cluster_{i}" for i in range(result["k"])]
                        if probabilities is not None else [])
        with (pending / "assignments.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow([*KEY, "cluster", *prob_columns])
            for i, (key, label) in enumerate(zip(keys, labels, strict=True)):
                writer.writerow([*key, int(label), *(probabilities[i] if prob_columns else ())])
    pending.rename(directory)


def run_search(train, reference, keys, output, manifest, *, budget_seconds, stop_requested):
    started = time.monotonic()
    completed = []
    final = []
    screening = {}
    halt = None

    def attempt(family, k, seed):
        nonlocal halt
        if stop_requested():
            halt = "signal"
            return None
        if time.monotonic() - started >= budget_seconds:
            halt = "time_budget"
            return None
        if len(completed) >= MAX_FITS:
            halt = "fit_budget"
            return None
        result, labels, probabilities = evaluate_fit(family, k, seed, train, reference)
        name = f"{family}-{k}-seed-{seed}"
        persist_fit(output / name, result, keys, labels, probabilities)
        completed.append(name)
        manifest["completed_fits"] = completed
        write_json(output / "manifest.json", manifest)
        return result, labels

    for family, k in CONFIGS:
        fit = attempt(family, k, SEEDS[0])
        if fit is None:
            break
        screening[family, k] = fit

    eligible = [
        {**result, "selection_silhouette": result["silhouette"]}
        for result, _ in screening.values() if result["status"] == "eligible"
    ]
    shortlist = candidate_order(eligible)[:3] if len(screening) == len(CONFIGS) else []
    for candidate in shortlist:
        family, k = candidate["family"], candidate["k"]
        attempts = [screening[family, k]]
        for seed in SEEDS[1:]:
            fit = attempt(family, k, seed)
            if fit is None:
                break
            attempts.append(fit)
        valid = len(attempts) == len(SEEDS) and all(r["status"] == "eligible" for r, _ in attempts)
        aris = [
            {"seeds": [left[0]["seed"], right[0]["seed"]],
             "ari": float(adjusted_rand_score(left[1], right[1]))}
            for left, right in itertools.combinations(attempts, 2)
            if left[1] is not None and right[1] is not None
        ]
        stable = valid and all(pair["ari"] >= MIN_ARI for pair in aris)
        final.append({
            "family": family, "k": k, "pairwise_ari": aris,
            "status": ("eligible" if stable else "incomplete" if len(attempts) < len(SEEDS)
                       else "rejected"),
            "reason": ("passed_all_seeds_and_ari" if stable else halt or
                       ("ari_below_threshold" if valid else "seed_failed_eligibility")),
            "selection_silhouette": min(r["silhouette"] for r, _ in attempts) if valid else None,
            "min_reference_cluster": min(r.get("min_reference_cluster", 0) for r, _ in attempts),
        })

    if halt is None and stop_requested():
        halt = "signal"
    ranked = candidate_order(item for item in final if item["status"] == "eligible")
    candidates = []
    for family, k in CONFIGS:
        screening_result = screening.get((family, k))
        candidates.append(screening_result[0] if screening_result else
                          {"family": family, "k": k, "status": "not_run", "reasons": [halt]})
    report = {
        "candidates": candidates, "finalists": final,
        "provisional_ranking": ranked, "selected": ranked[0] if ranked and halt is None else None,
        "attempted_fits": len(completed), "elapsed_seconds": time.monotonic() - started,
        "halt_reason": halt,
    }
    write_json(output / "report.json", report)
    manifest.update(status="stopped" if halt else "complete", halt_reason=halt,
                    search_seconds=report["elapsed_seconds"])
    write_json(output / "manifest.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split-seed", type=int, default=101)
    parser.add_argument("--budget-seconds", type=float, default=600)
    args = parser.parse_args()
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("--budget-seconds must be finite and positive")
    started = time.perf_counter()
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    directory = Path(__file__).resolve().parent / "samples" / "distinct_5k"
    source, dataset_id = load_manifest(directory)
    rows = build_snapshot(directory, expected_dataset_id=dataset_id)
    args.out.mkdir(parents=True)
    with threadpool_limits(limits=2):
        train, reference, keys, stats = prepare(rows, args.split_seed)
        manifest = {
            "status": "running", "dataset_id": dataset_id, "data_cutoff": source["cutoff"],
            "input_manifest": source,
            "source_manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
            "code_sha256": {name: hashlib.sha256((directory.parents[1] / name).read_bytes()).hexdigest()
                            for name in ("snapshot.py", "experiment.py")},
            "split_seed": args.split_seed, "model_seeds": SEEDS,
            "candidates": CONFIGS, "model_parameters": MODEL_PARAMS,
            "population": {"train": len(train), "reference": len(reference),
                           "reserved_test": len(rows) - len(train) - len(reference)},
            "reference_keys_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
            "evaluation_space": "P1 Euclidean, full fixed reference for silhouette and ARI",
            "fit_space": "P1 train, inertia and GMM AIC/BIC reported on train",
            "P1": stats, "fit_budget": MAX_FITS, "budget_seconds": args.budget_seconds,
            "numerical_threads": 2, "minimum_reference_cluster": max(30, math.ceil(0.03 * len(reference))),
            "silhouette_band": SILHOUETTE_BAND, "minimum_pairwise_ari": MIN_ARI,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {line.split("==")[0]: version(line.split("==")[0])
                             for line in (directory.parents[1] / "requirements.txt").read_text().splitlines()},
            "completed_fits": [],
        }
        write_json(args.out / "manifest.json", manifest)
        report = run_search(train, reference, keys, args.out, manifest,
                            budget_seconds=args.budget_seconds, stop_requested=lambda: stopping)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report.update(total_seconds=time.perf_counter() - started,
                  peak_rss_bytes=rss if sys.platform == "darwin" else rss * 1024)
    write_json(args.out / "report.json", report)
    print(f"{report['attempted_fits']} fits; status={manifest['status']}; report={args.out / 'report.json'}")


if __name__ == "__main__":
    main()
