"""M2.2B preregistered three-phase validation."""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
from importlib.metadata import version
import signal
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics.cluster import contingency_matrix
from threadpoolctl import threadpool_limits

from controls import (BINARY, CANDIDATES, USAGE, VALUE, prepare, sha256, stability_report)
from experiment import MODEL_PARAMS, evaluate_fit, feature_array, persist_fit, write_json
from sensitivity import (INIT_SEEDS, SCREEN_SEED, SPLIT_SEED, SUBSAMPLE_SEEDS,
                         characterize, digest, rss_bytes, subsample)
from snapshot import FEATURES, KEY, build_snapshot, load_manifest, read_table, split_snapshot

ROOT = Path(__file__).resolve().parent
SCENARIOS = ("distinct", "overlap", "continuum")
MODELS = ("C1", "C3", "C4")
RULES = ("zero", "low_use", "high_use", "coverage", "value", "value_coverage")
OUTCOMES = ("renovou_90d", "adotou_produto_90d", "queda_uso_90d")
FROZEN = ("PROTOCOLO.md", "validation.py", "snapshot.py", "experiment.py",
          "sensitivity.py", "controls.py", "requirements.txt")
MAX_FITS = 100
MAX_SECONDS = 600


def now():
    return datetime.now(timezone.utc).isoformat()


def hashes():
    return {name: sha256(ROOT / name) for name in FROZEN}


def load(path):
    return json.loads(path.read_text())


def check_digest(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Frozen artifact changed: {path.name}")


def key(row):
    return tuple(row[field] for field in KEY)


def population(scenario, *parts):
    source = ROOT / "samples" / f"{scenario}_5k"
    manifest, dataset_id = load_manifest(source)
    rows = build_snapshot(source, expected_dataset_id=dataset_id)
    splits = {key(row): row["split"] for row in split_snapshot(rows, seed=SPLIT_SEED)}
    return source, manifest, {part: [row for row in rows if splits[key(row)] == part] for part in parts}


def outcomes(source, rows):
    manifest, dataset_id = load_manifest(source)
    keys = [key(row) for row in rows]
    cutoff = manifest["cutoff"]
    if (len(set(keys)) != len(keys)
            or any((k[0], k[2]) != (dataset_id, cutoff) for k in keys)):
        raise ValueError("Outcome join requires one dataset, cutoff and unique client keys")
    wanted = {k[1] for k in keys}
    accepted = {}
    # The gzip contains all partitions. Interpret outcome fields only for requested IDs.
    for row in read_table(source, manifest, "futuro_reservado"):
        client = row["cliente_id"]
        if client in wanted:
            if row["janela_fim"] != manifest["forward_window_end"]:
                raise ValueError("Unexpected outcome window")
            if any(row[field] not in {"0", "1"} for field in OUTCOMES):
                raise ValueError("Invalid outcome")
            accepted[client] = [int(row[field]) for field in OUTCOMES]
    if accepted.keys() != wanted:
        raise ValueError("Incomplete outcome keys")
    return np.asarray([accepted[k[1]] for k in keys])


def rules(matrix, value_cutoff):
    value, usage, cov = (matrix[:, FEATURES.index(field)] for field in (VALUE, USAGE, BINARY))
    band = (value >= value_cutoff).astype(int)
    return {"zero": (value == 0).astype(int),
            "low_use": ((cov == 1) & (usage <= 7)).astype(int),
            "high_use": ((cov == 1) & (usage >= 18)).astype(int),
            "coverage": cov.astype(int), "value": band,
            "value_coverage": 2 * band + cov.astype(int)}


def apply_frozen(raw, stats, centers):
    data = {field: raw[:, col].copy() for col, field in enumerate(FEATURES)}
    for field in stats.get("log1p", ()):
        data[field] = np.log1p(data[field])
    for field, (lower, upper) in stats["limits"].items():
        data[field] = np.clip(data[field], lower, upper)
    for field, median in stats["imputer_medians"].items():
        values = data[field]
        data[field] = ((np.where(np.isnan(values), median, values)
                       - stats["scaler_center"][field]) / stats["scaler_scale"][field])
    values = np.column_stack([data[field] for field in stats["model_features"]])
    return np.argmin(((values[:, None] - np.asarray(centers)[None, :]) ** 2).sum(axis=2), axis=1)


def train_all(out, budget, stopped, *, recover=None):
    started = time.monotonic()
    run = {"frozen_at": now(), "hashes": hashes(), "fit_budget": MAX_FITS, "seconds_budget": budget,
           "dependencies": {line.split("==")[0]: version(line.split("==")[0])
                            for line in (ROOT / "requirements.txt").read_text().splitlines()},
           "scenario": {}, "completed": [], "parameters": MODEL_PARAMS["kmeans"],
           "split_seed": SPLIT_SEED, "screen_seed": SCREEN_SEED, "subsample_seeds": SUBSAMPLE_SEEDS,
           "initialization_seeds": INIT_SEEDS, "refit_policy": "none"}
    prior = {}
    if recover is not None:
        old = load(recover / "fit_manifest.json")
        if (recover / "reference_started.json").exists() or (ROOT / "TESTE_ABERTO.json").exists():
            raise ValueError("Pre-outcome recovery is no longer allowed")
        if any(old["hashes"][name] != run["hashes"][name] for name in FROZEN
               if name not in {"validation.py", "PROTOCOLO.md"}):
            raise ValueError("Model implementation or dependencies changed")
        prior = {name: recover / name for name in old["completed"]}
        run["recovery"] = {"prior_manifest_sha256": sha256(recover / "fit_manifest.json"),
                           "prior_preregistration_sha256": sha256(recover / "preregistration.json"),
                           "reused_fits": len(prior), "unpersisted_failed_attempts": 1}
    out.mkdir(parents=True)
    write_json(out / "preregistration.json", run)
    for scenario in SCENARIOS:
        source, _, populations = population(scenario, "train", "reference")
        train, ref = (feature_array(populations[p]) for p in ("train", "reference"))
        scenario_dir = out / scenario
        scenario_dir.mkdir()
        keys = [key(row) for row in populations["reference"]]
        train_keys = [key(row) for row in populations["train"]]
        samples = {seed: subsample(len(train), seed) for seed in SUBSAMPLE_SEEDS}
        record = {"source_sha256": sha256(source / "manifest.json"), "dataset_id": keys[0][0],
                  "reference_keys_sha256": digest(keys), "train_keys_sha256": digest(train_keys),
                  "value_cutoff": float(np.median(train[:, FEATURES.index(VALUE)])),
                  "sample_key_hashes": {seed: digest([train_keys[i] for i in idx])
                                        for seed, idx in samples.items()}, "models": {}}
        run["scenario"][scenario] = record
        for code in MODELS:
            fits = []
            for sample_seed, seed in [(None, SCREEN_SEED), *itertools.product(SUBSAMPLE_SEEDS, INIT_SEEDS)]:
                if stopped() or time.monotonic() - started >= budget or len(run["completed"]) + bool(recover) >= MAX_FITS:
                    run.update(status="stopped", seconds=time.monotonic() - started, peak_rss_bytes=rss_bytes())
                    write_json(out / "fit_manifest.json", run)
                    return run
                name = f"{code}-{sample_seed or 'initial'}-{seed}"
                artifact = f"{scenario}/{name}"
                if artifact in prior:
                    if (old["scenario"][scenario]["reference_keys_sha256"] != digest(keys)
                            or old["scenario"][scenario]["source_sha256"] != record["source_sha256"]
                            or old["scenario"][scenario]["train_keys_sha256"] != digest(train_keys)):
                        raise ValueError("Recovered fit does not match the frozen populations")
                    shutil.copytree(prior[artifact], scenario_dir / name)
                    metrics = load(scenario_dir / name / "metrics.json")
                    with (scenario_dir / name / "assignments.csv").open(newline="") as stream:
                        assignments = list(csv.DictReader(stream))
                    if [key(row) for row in assignments] != keys:
                        raise ValueError("Recovered assignment keys changed")
                    labels = np.array([int(row["cluster"]) for row in assignments])
                else:
                    raw = train if sample_seed is None else train[samples[sample_seed]]
                    tx, rx, stats = prepare(code, raw, ref)
                    metrics, labels, _ = evaluate_fit("kmeans", 2, seed, tx, rx, include_centers=True)
                    metrics.update(artifact=name, subsample_seed=sample_seed, preprocessing=stats)
                    if sample_seed is None and labels is not None:
                        np.testing.assert_array_equal(apply_frozen(ref, stats, metrics["centers"]), labels)
                        metrics["profiles"] = characterize(ref, labels, 2, np.zeros(len(ref), dtype=bool))
                    persist_fit(scenario_dir / name, metrics, keys, labels, None)
                run["completed"].append(artifact)
                fits.append((metrics, labels))
                write_json(out / "fit_manifest.json", run)
            initial, labels = fits[0]
            folder = scenario_dir / initial["artifact"]
            record["models"][code] = {
                "artifact": initial["artifact"], "status": initial["status"],
                "metrics_sha256": sha256(folder / "metrics.json"),
                "assignments_sha256": sha256(folder / "assignments.csv") if labels is not None else None,
                "stability": stability_report(labels, fits[1:]) if labels is not None else None,
            }
    status = "stopped" if stopped() or time.monotonic() - started >= budget else "complete"
    run.update(status=status, seconds=time.monotonic() - started, peak_rss_bytes=rss_bytes())
    write_json(out / "fit_manifest.json", run)
    return run


def group_summary(labels, y, seed=4021, repeats=2000):
    rng = np.random.default_rng(seed)
    tab = contingency_matrix(labels, y)
    v = (float(np.sqrt(chi2_contingency(tab, correction=False)[0] / len(y)))
         if min(tab.shape) > 1 else None)
    groups = []
    for label in sorted(set(labels)):
        inside = y[labels == label]
        outside = y[labels != label]
        rate = float(inside.mean())
        result = {"group": int(label), "n": len(inside), "events": int(inside.sum()), "rate": rate}
        if len(outside):
            other = float(outside.mean())
            # For binary observations this is exactly resampling clients within each group.
            diff = rng.binomial(len(inside), rate, repeats) / len(inside)
            diff -= rng.binomial(len(outside), other, repeats) / len(outside)
            result.update(difference_pp=100 * (rate - other), risk_ratio=rate / other if other else None,
                          ci95_pp=(100 * np.quantile(diff, [.025, .975])).tolist())
        groups.append(result)
    return {"cramers_v": v, "groups": groups}


def cells(labels, fields):
    return list(zip(*(labels[field] for field in fields), strict=True))


def cell_rates(keys, y):
    counts = {}
    for cell, event in zip(keys, y, strict=True):
        hits, n = counts.get(cell, (0, 0))
        counts[cell] = hits + int(event), n + 1
    return {cell: (hits + 1) / (n + 2) for cell, (hits, n) in counts.items()}


def prediction(train_groups, train_y, eval_groups, method):
    base_fields = RULES if method in MODELS else ()
    global_rate = (train_y.sum() + 1) / (len(train_y) + 2)
    if base_fields:
        rates = cell_rates(cells(train_groups, base_fields), train_y)
        base = [rates.get(cell, global_rate) for cell in cells(eval_groups, base_fields)]
    else:
        base = [global_rate] * len(eval_groups[method])
    extended = (*base_fields, method)
    rates = cell_rates(cells(train_groups, extended), train_y)
    better = [rates.get(cell, fallback) for cell, fallback in zip(cells(eval_groups, extended), base, strict=True)]
    return np.asarray(base), np.asarray(better)


def folds(keys):
    order = sorted(range(len(keys)), key=lambda i: digest([4022, keys[i]]))
    result = np.empty(len(keys), dtype=int)
    result[order] = np.arange(len(keys)) % 5
    return result


def brier(groups, y, method, partition, *, calibration=None):
    if calibration is None:
        base, better = np.empty(len(y)), np.empty(len(y))
        for fold in range(5):
            idx = partition == fold
            base[idx], better[idx] = prediction(
                {name: values[~idx] for name, values in groups.items()}, y[~idx],
                {name: values[idx] for name, values in groups.items()}, method)
    else:
        base, better = prediction(*calibration, groups, method)
    base_loss, better_loss = (base - y) ** 2, (better - y) ** 2
    diff = base_loss - better_loss
    rng = np.random.default_rng(4023)
    estimates = np.concatenate([diff[rng.integers(len(y), size=(500, len(y)))].mean(axis=1)
                                for _ in range(20)])
    alpha = .05 / (2 * 27)
    low, high = np.quantile(estimates, [alpha, 1 - alpha]).tolist()
    gain = float(diff.mean())
    return {"baseline": float(base_loss.mean()), "candidate": float(better_loss.mean()),
            "gain": gain, "adjusted_ci": [low, high], "passes": gain >= .002 and low > 0}


def matched(raw, model, y):
    rng = np.random.default_rng(4021)
    value = raw[:, FEATURES.index(VALUE)]
    usage = raw[:, FEATURES.index(USAGE)]
    controls = {"lowest_value": (value, 1), "highest_value": (value, -1),
                "lowest_usage": (usage, 1), "highest_usage": (usage, -1)}
    result = {}
    for group in sorted(set(model)):
        member = model == group
        n = int(sum(member))
        variants = {}
        for name, (values, direction) in controls.items():
            observed = np.flatnonzero(np.isfinite(values))
            if len(observed) < n:
                variants[name] = {"status": "insufficient_observed_clients", "available": len(observed)}
                continue
            order = observed[np.argsort(direction * values[observed], kind="stable")]
            control = np.zeros(len(y), dtype=bool)
            control[order[:n]] = True
            # Paired customer resampling including overlapping memberships.
            categories, counts = np.unique(np.column_stack([member, control, y]), axis=0, return_counts=True)
            frequency = rng.multinomial(len(y), counts / len(y), size=2000)
            left, right, event = categories.T
            denom_left, denom_right = frequency @ left, frequency @ right
            valid = (denom_left > 0) & (denom_right > 0)
            diff = ((frequency[valid] @ (left * event)) / denom_left[valid]
                    - (frequency[valid] @ (right * event)) / denom_right[valid])
            variants[name] = {"n_each": n, "overlap": int(sum(member & control)),
                              "model_events": int(y[member].sum()), "rule_events": int(y[control].sum()),
                              "difference_pp": float(100 * (y[member].mean() - y[control].mean())),
                              "ci95_pp": (100 * np.quantile(diff, [.025, .975])).tolist() if len(diff) else None}
        result[int(group)] = variants
    return result


def validate_fit(out):
    fitted = load(out / "fit_manifest.json")
    prereg = load(out / "preregistration.json")
    if fitted["status"] != "complete" or len(fitted["completed"]) != 99:
        raise ValueError("A complete 99-fit run is required")
    if fitted["hashes"] != hashes() or fitted["hashes"] != prereg["hashes"]:
        raise ValueError("Code or protocol differs from preregistration")
    if any(version(name) != expected for name, expected in fitted["dependencies"].items()):
        raise ValueError("Dependency versions changed")
    for scenario, record in fitted["scenario"].items():
        source = ROOT / "samples" / f"{scenario}_5k"
        check_digest(source / "manifest.json", record["source_sha256"])
        for model in record["models"].values():
            folder = out / scenario / model["artifact"]
            check_digest(folder / "metrics.json", model["metrics_sha256"])
            if model["assignments_sha256"]:
                check_digest(folder / "assignments.csv", model["assignments_sha256"])
    return fitted


def groups_for(scenario, rows, record, out, part):
    raw = feature_array(rows)
    result = rules(raw, record["value_cutoff"])
    for code, model in record["models"].items():
        folder = out / scenario / model["artifact"]
        if not model["assignments_sha256"]:
            continue
        if part == "reference":
            with (folder / "assignments.csv").open(newline="") as stream:
                assignments = list(csv.DictReader(stream))
            if [key(row) for row in assignments] != [key(row) for row in rows]:
                raise ValueError("Reference assignment keys differ")
            result[code] = np.array([int(row["cluster"]) for row in assignments])
        else:
            metrics = load(folder / "metrics.json")
            result[code] = apply_frozen(raw, metrics["preprocessing"], metrics["centers"])
    return raw, result


def evaluate_population(source, rows, groups, raw, *, calibration=None):
    if calibration is not None and {key(row) for row in rows}.intersection(calibration["keys"]):
        raise ValueError("Calibration and confirmation overlap")
    y = outcomes(source, rows)
    partition = folds([key(row) for row in rows]) if calibration is None else None
    report = {}
    for i, field in enumerate(OUTCOMES):
        events = y[:, i]
        calibrated = None if calibration is None else (calibration["groups"], calibration["y"][:, i])
        report[field] = {
            "events": int(events.sum()),
            "methods": {method: {
                **group_summary(labels, events),
                "brier": brier(groups, events, method, partition, calibration=calibrated),
                **({"size_matched": matched(raw, labels, events)} if method in MODELS else {}),
            } for method, labels in groups.items()},
        }
    return report


def select(reference, fitted):
    options = []
    for method in (*MODELS, *RULES):
        if method in MODELS and not all(
            fitted["scenario"][s]["models"][method]["status"] == "eligible"
            and fitted["scenario"][s]["models"][method]["stability"]["all_runs"]["passes_stability_gate"]
            for s in ("distinct", "overlap")
        ):
            continue
        for target in OUTCOMES:
            scores = [reference[s][target]["methods"][method]["brier"] for s in ("distinct", "overlap")]
            if all(score["passes"] for score in scores):
                options.append((method not in MODELS, -min(score["gain"] for score in scores), method, target))
    if not options:
        return {"method": None, "outcome": None, "decision": "no_method"}
    _, _, method, field = min(options)
    return {"method": method, "outcome": field, "decision": "model" if method in MODELS else "rule"}


def claim(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


def reference_phase(out):
    fitted = validate_fit(out)
    claim(out / "reference_started.json", {"started_at": now(), "hashes": hashes()})
    report = {}
    for scenario in SCENARIOS:
        source, _, pops = population(scenario, "reference")
        rows = pops["reference"]
        record = fitted["scenario"][scenario]
        if digest([key(row) for row in rows]) != record["reference_keys_sha256"]:
            raise ValueError("Reference keys changed")
        raw, groups = groups_for(scenario, rows, record, out, "reference")
        report[scenario] = evaluate_population(source, rows, groups, raw)
        write_json(out / f"{scenario}-reference.json", report[scenario])
    decision = select(report, fitted)
    freeze = {"frozen_at": now(), "hashes": hashes(), "selection": decision,
              "candidates": [*MODELS, *RULES], "refit_policy": "no model, preprocessing or rule refit",
              "preregistration_sha256": sha256(out / "preregistration.json"),
              "fit_manifest_sha256": sha256(out / "fit_manifest.json"),
              "reference_sha256": {s: sha256(out / f"{s}-reference.json") for s in SCENARIOS}}
    claim(out / "confirmation_freeze.json", freeze)
    return decision


def confirm_phase(out):
    frozen = load(out / "confirmation_freeze.json")
    fitted = validate_fit(out)
    if frozen["hashes"] != hashes():
        raise ValueError("Frozen code changed")
    check_digest(out / "preregistration.json", frozen["preregistration_sha256"])
    check_digest(out / "fit_manifest.json", frozen["fit_manifest_sha256"])
    reference = {}
    for scenario, expected in frozen["reference_sha256"].items():
        path = out / f"{scenario}-reference.json"
        check_digest(path, expected)
        reference[scenario] = load(path)
    decision = select(reference, fitted)
    if decision != frozen["selection"]:
        raise ValueError("Frozen selection differs")
    opening = {"opened_at": now(), "run": str(out.resolve()),
               "freeze_sha256": sha256(out / "confirmation_freeze.json")}
    claim(ROOT / "TESTE_ABERTO.json", opening)
    claim(out / "test_opened.json", opening)
    summary = {"selection": decision, "confirmation": "no_selection_to_confirm"}
    for scenario in SCENARIOS:
        source, _, pops = population(scenario, "reference", "test")
        record = fitted["scenario"][scenario]
        ref, test = pops["reference"], pops["test"]
        test_keys = [key(row) for row in test]
        ref_keys = [key(row) for row in ref]
        if digest(ref_keys) != record["reference_keys_sha256"] or set(ref_keys) & set(test_keys):
            raise ValueError("Invalid test independence")
        _, ref_groups = groups_for(scenario, ref, record, out, "reference")
        raw, groups = groups_for(scenario, test, record, out, "test")
        calibration = {"groups": ref_groups, "y": outcomes(source, ref), "keys": ref_keys}
        result = evaluate_population(source, test, groups, raw, calibration=calibration)
        write_json(out / f"{scenario}-test.json", {"keys_sha256": digest(test_keys), "results": result})
        if decision["method"] and scenario != "continuum":
            trial = result[decision["outcome"]]["methods"][decision["method"]]["brier"]
            summary.setdefault("selected_test", {})[scenario] = trial
    if "selected_test" in summary:
        summary["confirmation"] = ("confirmed" if all(x["passes"] for x in summary["selected_test"].values())
                                   else "not_confirmed")
    write_json(out / "confirmation.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "reference", "confirm"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.add_argument("--recover", type=Path)
    args = parser.parse_args()
    if args.recover and args.phase != "fit":
        parser.error("--recover applies to fit only")
    if not 0 < args.budget_seconds <= MAX_SECONDS:
        parser.error("budget must be in (0, 600]")
    stopped = False

    def request_stop(_number, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    start = time.monotonic()
    with threadpool_limits(limits=2):
        if args.phase == "fit":
            run = train_all(args.out, args.budget_seconds, lambda: stopped, recover=args.recover)
            print(f"{run['status']}: {len(run['completed'])} fits")
        else:
            result = reference_phase(args.out) if args.phase == "reference" else confirm_phase(args.out)
            print(result)
    print(f"{args.phase}: {time.monotonic() - start:.2f}s, peak RSS {rss_bytes() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
