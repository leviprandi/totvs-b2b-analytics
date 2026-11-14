"""M2.1 sensitivity and resampling on the fixed M1 train/reference split."""
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
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from experiment import (CONFIGS, MODEL_PARAMS, NUMERIC, candidate_order, evaluate_fit,
                        feature_array, fit_p1, partition_rows, persist_fit, write_json)
from snapshot import FEATURES, KEY, build_snapshot, load_manifest


ROOT = Path(__file__).resolve().parent
REPRESENTATIONS = ("P1", "P2", "P1_clip")
LOG_FIELDS = ("valor_mensal_total", "qtd_contratos", "qtd_produtos",
              "dias_relacionamento", "qtd_eventos_uso")
CLIP_FIELDS = ("valor_mensal_total", "qtd_eventos_uso")
QUANTILES = (0.005, 0.995)
SCREEN_SEED = 11
SUBSAMPLE_SEEDS = (1011, 1012, 1013, 1014, 1015)
INIT_SEEDS = (11, 29)
SPLIT_SEED = 101
MAX_SCREEN, MAX_RESAMPLE = 42, 40


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def prepare_representation(name, train, reference):
    train, reference = train.copy(), reference.copy()
    limits = {}
    if name == "P2":
        columns = [FEATURES.index(field) for field in LOG_FIELDS]
        for matrix in (train, reference):
            values = matrix[:, columns]
            if np.any(values < 0):
                raise ValueError("P2 log1p inputs must be nonnegative")
            matrix[:, columns] = np.log1p(values)
    elif name == "P1_clip":
        for field in CLIP_FIELDS:
            column = FEATURES.index(field)
            values = train[:, column]
            if np.isnan(values).all():
                raise ValueError(f"No observed values for clipping {field}")
            lo, hi = np.nanquantile(values, QUANTILES).tolist()
            limits[field] = [lo, hi]
            for matrix in (train, reference):
                matrix[:, column] = np.clip(matrix[:, column], lo, hi)
    elif name != "P1":
        raise ValueError(f"Unknown representation: {name}")
    train, reference, stats = fit_p1(train, reference)
    stats.update(log1p=list(LOG_FIELDS) if name == "P2" else [], limits=limits)
    return train, reference, stats


def subsample(n_rows, seed):
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=n_rows * 80 // 100,
                                                     replace=False))


def shortlist(screen):
    """Reserve up to one slot per geometry, then fill the fourth by family diversity."""
    queues = {}
    for rep in REPRESENTATIONS:
        eligible = [{**result, "selection_silhouette": result["silhouette"]}
                    for (r, _, _), (result, _) in screen.items()
                    if r == rep and result["status"] == "eligible"]
        queues[rep] = candidate_order(eligible)

    chosen = []
    for rep, queue in queues.items():
        if queue:
            families = {family for _, family, _ in chosen}
            pick = next((item for item in queue if item["family"] not in families), queue[0])
            chosen.append((rep, pick["family"], pick["k"]))
    leftovers = [(rank, rep, item) for rep, queue in queues.items()
                 for rank, item in enumerate(queue)
                 if (rep, item["family"], item["k"]) not in chosen]
    while len(chosen) < 4 and leftovers:
        counts = {family: sum(key[1] == family for key in chosen) for family in MODEL_PARAMS}
        _, rep, item = min(leftovers, key=lambda entry: (
            counts[entry[2]["family"]], entry[0], REPRESENTATIONS.index(entry[1])))
        chosen.append((rep, item["family"], item["k"]))
        leftovers = [entry for entry in leftovers if not (entry[1] == rep and entry[2] is item)]
    return chosen


def characterize(raw, labels, k, tail):
    profiles = []
    for cluster in range(k):
        member = labels == cluster
        block = raw[member]
        summaries = {}
        for field in FEATURES:
            values = block[:, FEATURES.index(field)]
            observed = values[~np.isnan(values)]
            summaries[field] = {
                "observed": int(len(observed)),
                "median": float(np.median(observed)) if len(observed) else None,
                "mean": float(np.mean(observed)) if len(observed) else None,
            }
        profiles.append({"cluster": cluster, "size": int(member.sum()),
                         "outside_train_cutoffs": int((member & tail).sum()),
                         "features": summaries})
    return profiles


def tail_diagnostic(train, reference, bounds):
    fields = {}
    flags = []
    for matrix in (train, reference):
        outside = np.zeros(len(matrix), dtype=bool)
        for field, (lo, hi) in bounds.items():
            values = matrix[:, FEATURES.index(field)]
            mask = (values < lo) | (values > hi)
            fields.setdefault(field, []).append(int(mask.sum()))
            outside |= mask
        flags.append(outside)
    return flags, {"counts_train_reference_by_field": fields,
                   "unique_train_reference": [int(mask.sum()) for mask in flags]}


def summarize_stability(baseline, runs, k):
    successful = [(run, labels) for run, labels in runs if labels is not None]
    pairs = [{"left": a["artifact"], "right": b["artifact"],
              "ari": float(adjusted_rand_score(left, right)),
              "same_subsample": a["subsample_seed"] == b["subsample_seed"]}
             for (a, left), (b, right) in itertools.combinations(successful, 2)]
    scores = np.asarray([p["ari"] for p in pairs])
    cross_sample = np.asarray([p["ari"] for p in pairs if not p["same_subsample"]])
    aligned = []
    for _, labels in successful:
        counts = np.zeros((k, k), dtype=int)
        np.add.at(counts, (baseline, labels), 1)
        row, col = linear_sum_assignment(counts, maximize=True)
        fractions = np.zeros(k)
        fractions[row] = counts.sum(axis=0)[col] / len(baseline)
        aligned.append(fractions)
    groups = []
    if aligned:
        for i, shares in enumerate(np.asarray(aligned).T):
            p10, p90 = np.quantile(shares, (0.1, 0.9))
            groups.append({"cluster": i, "share_p10": float(p10), "share_p90": float(p90),
                           "spread_percentage_points": float(100 * (p90 - p10)),
                           "absent_runs": int((shares == 0).sum())})

    def summary(values):
        return ({"pairs": len(values), "median": float(np.median(values)),
                 "p10": float(np.quantile(values, 0.1)), "std": float(values.std())}
                if len(values) else {"pairs": 0, "median": None, "p10": None, "std": None})

    eligible = sum(run["status"] == "eligible" for run, _ in runs)
    ari = summary(cross_sample)
    reasons = []
    if len(runs) < 10:
        reasons.append("incomplete")
    if eligible < 9:
        reasons.append("fewer_than_9_eligible_runs")
    if ari["median"] is None or ari["median"] < .75 or ari["p10"] < .60:
        reasons.append("cross_sample_ari_below_floor")
    if any(group["spread_percentage_points"] > 10 + 1e-9 or group["absent_runs"]
           for group in groups):
        reasons.append("cluster_share_variation_or_disappearance")
    return {"runs": len(runs), "eligible_runs": eligible, "successful_runs": len(successful),
            "ari_all": summary(scores), "ari_between_subsamples": ari, "pairs": pairs,
            "aligned_group_shares": groups, "passes_stability_gate": not reasons, "reasons": reasons}


def run(train, reference, train_keys, ref_keys, out, manifest, *, budget, stopped):
    started = time.monotonic()
    completed = []
    screen = {}
    finalists = []
    halt = None

    def can_start():
        nonlocal halt
        if stopped():
            halt = "signal"
        elif time.monotonic() - started >= budget:
            halt = "time_budget"
        return halt is None

    def fit(rep, family, k, seed, prepared, sample_seed=None, indices=None):
        tx, rx, stats = prepared
        result, labels, probabilities = evaluate_fit(family, k, seed, tx, rx,
                                                     include_train_labels=indices is None)
        phase = "screen" if indices is None else "resample"
        name = f"{rep}-{family}-{k}-seed-{seed}"
        if sample_seed is not None:
            name += f"-sample-{sample_seed}"
        result.update(representation=rep, phase=phase, artifact=name, subsample_seed=sample_seed,
                      train_rows=len(tx), reference_rows=len(rx), preprocessing=stats,
                      peak_process_rss_bytes=rss_bytes(),
                      silhouette_space=f"{rep} Euclidean on fixed reference, fit to current train")
        train_labels = result.pop("train_labels", None)
        if labels is not None:
            result.update(reference_assignment_coverage=len(labels) / len(reference),
                          effective_train_groups=sum(c["train_count"] > 0 for c in result["clusters"]),
                          effective_reference_groups=sum(c["reference_count"] > 0 for c in result["clusters"]),
                          vanished_groups=[c["cluster"] for c in result["clusters"]
                                           if c["train_count"] and not c["reference_count"]])
        if train_labels is not None:
            result["train_profiles"] = characterize(train, np.asarray(train_labels), k, train_tail)
            result["reference_profiles"] = characterize(reference, labels, k, ref_tail)
        persist_fit(out / name, result, ref_keys, labels, probabilities)
        completed.append(name)
        manifest["completed_fits"] = completed
        write_json(out / "manifest.json", manifest)
        return result, labels

    prepared = {rep: prepare_representation(rep, train, reference) for rep in REPRESENTATIONS}
    (train_tail, ref_tail), tails = tail_diagnostic(train, reference, prepared["P1_clip"][2]["limits"])
    all_configs = [(rep, family, k) for rep in REPRESENTATIONS for family, k in CONFIGS]
    for rep, family, k in all_configs[:MAX_SCREEN]:
        if not can_start():
            break
        screen[rep, family, k] = fit(rep, family, k, SCREEN_SEED, prepared[rep])

    baseline = {}
    price = FEATURES.index("valor_mensal_total")
    for k in range(2, 9):
        edges = np.unique(np.quantile(train[:, price], np.arange(1, k) / k))
        baseline[k] = (edges, np.searchsorted(edges, reference[:, price], side="right"))

    cross = []
    for family, k in CONFIGS:
        for left, right in itertools.combinations(REPRESENTATIONS, 2):
            a = screen.get((left, family, k), (None, None))[1]
            b = screen.get((right, family, k), (None, None))[1]
            if a is not None and b is not None:
                cross.append({"family": family, "k": k, "representations": [left, right],
                              "ari": float(adjusted_rand_score(a, b))})

    sample_indices = {seed: subsample(len(train), seed) for seed in SUBSAMPLE_SEEDS}
    manifest["subsamples"] = {seed: {"rows": len(idx),
                                    "keys_sha256": digest([train_keys[i] for i in idx])}
                              for seed, idx in sample_indices.items()}
    count_resamples = 0
    for rep, family, k in shortlist(screen) if len(screen) == len(all_configs) and halt is None else []:
        runs = []
        for sample_seed, seed in itertools.product(SUBSAMPLE_SEEDS, INIT_SEEDS):
            if count_resamples >= MAX_RESAMPLE or not can_start():
                break
            idx = sample_indices[sample_seed]
            transformed = prepare_representation(rep, train[idx], reference)
            runs.append(fit(rep, family, k, seed, transformed, sample_seed, idx))
            count_resamples += 1
        initial = screen[rep, family, k][1]
        finalists.append({"representation": rep, "family": family, "k": k,
                          **summarize_stability(initial, runs, k)})
        if halt:
            break

    if stopped():
        halt = "signal"
    elif time.monotonic() - started >= budget and halt is None:
        halt = "time_budget"

    matrix = []
    for rep, family, k in all_configs:
        result, labels = screen.get((rep, family, k), (
            {"representation": rep, "family": family, "k": k, "status": "not_run"}, None))
        row = {key: result.get(key) for key in (
            "representation", "family", "k", "status", "reasons", "converged", "silhouette",
            "min_reference_cluster", "vanished_groups", "effective_train_groups",
            "effective_reference_groups", "reference_assignment_coverage", "inertia_train",
            "aic_train", "bic_train", "fit_seconds", "peak_process_rss_bytes", "artifact")}
        row["ari_value_bands"] = (float(adjusted_rand_score(labels, baseline[k][1]))
                                  if labels is not None else None)
        matrix.append(row)
    with (out / "screen.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    report = {
        "screen": matrix, "finalists": finalists, "cross_representation_ari": cross,
        "tail_diagnostic": tails, "value_bands": {
            k: {"edges_from_train": edges.tolist(), "reference_counts": np.bincount(labels).tolist(),
                "reference_profiles": characterize(reference, labels, len(edges) + 1, ref_tail)}
            for k, (edges, labels) in baseline.items()},
        "screen_fits": len(screen), "resample_fits": count_resamples, "halt_reason": halt,
        "robust_candidates": [
            [item["representation"], item["family"], item["k"]]
            for item in finalists if item["passes_stability_gate"]] if halt is None else [],
        "elapsed_seconds": time.monotonic() - started, "peak_process_rss_bytes": rss_bytes(),
    }
    write_json(out / "report.json", report)
    manifest.update(status="stopped" if halt else "complete", halt_reason=halt)
    write_json(out / "manifest.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--budget-seconds", type=float, default=600)
    args = parser.parse_args()
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("budget must be finite and positive")
    started = time.perf_counter()
    stopping = False

    def stop(_number, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    path = ROOT / "samples" / "distinct_5k"
    source, dataset_id = load_manifest(path)
    rows = build_snapshot(path, expected_dataset_id=dataset_id)
    train_rows, ref_rows = partition_rows(rows, SPLIT_SEED)
    train, reference = feature_array(train_rows), feature_array(ref_rows)
    train_keys, ref_keys = ([tuple(row[field] for field in KEY) for row in items]
                            for items in (train_rows, ref_rows))
    manifest = {
        "status": "running", "dataset_id": dataset_id, "source_manifest": source,
        "source_manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
        "split_seed": SPLIT_SEED, "train_keys_sha256": digest(train_keys),
        "reference_keys_sha256": digest(ref_keys), "population": {
            "train": len(train), "reference": len(reference), "reserved": len(rows)-len(train)-len(reference)},
        "features": FEATURES, "representation_order": REPRESENTATIONS,
        "screen_seed": SCREEN_SEED, "subsample_seeds": SUBSAMPLE_SEEDS, "init_seeds": INIT_SEEDS,
        "subsample_fraction": .8, "with_replacement": False, "configurations": CONFIGS,
        "model_parameters": MODEL_PARAMS, "clip_quantiles": QUANTILES,
        "evaluation": "Silhouette only within same fitted space; ARI on fixed reference keys",
        "aic_bic": "Compare only GMMs in the same representation and same fitted training population",
        "minimum_reference_cluster": max(30, math.ceil(.03 * len(reference))),
        "stability_gate": {"eligible_runs": 9, "ari_median": .75, "ari_p10": .60,
                           "max_share_spread_pp": 10},
        "budget": {"screen_fits": MAX_SCREEN, "resample_fits": MAX_RESAMPLE,
                   "seconds": args.budget_seconds}, "threads": 2,
        "dependencies": {line.split("==")[0]: version(line.split("==")[0])
                         for line in (ROOT / "requirements.txt").read_text().splitlines()},
        "python": platform.python_version(), "platform": platform.platform(),
        "code_sha256": {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest()
                        for file in ("sensitivity.py", "experiment.py", "snapshot.py")},
        "completed_fits": [],
    }
    args.out.mkdir(parents=True)
    write_json(args.out / "manifest.json", manifest)
    with threadpool_limits(limits=2):
        report = run(train, reference, train_keys, ref_keys, args.out, manifest,
                     budget=args.budget_seconds, stopped=lambda: stopping)
    report["total_seconds"] = time.perf_counter() - started
    write_json(args.out / "report.json", report)
    print(f"{report['screen_fits']} screen, {report['resample_fits']} resample; {manifest['status']}")


if __name__ == "__main__":
    main()
