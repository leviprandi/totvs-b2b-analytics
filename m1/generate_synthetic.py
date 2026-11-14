#!/usr/bin/env python3
"""Standalone, original synthetic B2B data generator for TOTVS 2.0 (draft).

Requires numpy and pandas. Makes no calls to the network, has no connection to
TOTVS data, and produces records in bounded chunks. Not a validated simulator.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "0.1.0"
CUTOFF = "2026-09-30"
SCENARIOS = ("distinct", "overlap", "continuum")
SECTORS = np.array(["SERVICOS_FICTICIOS", "VAREJO_FICTICIO", "INDUSTRIA_FICTICIA", "EDUCACAO_FICTICIA", "SAUDE_FICTICIA"])
REGIONS = np.array(["REGIAO_A", "REGIAO_B", "REGIAO_C", "REGIAO_D"])
PRODUCTS = np.array(["PRODUTO_FICTICIO_A", "PRODUTO_FICTICIO_B", "PRODUTO_FICTICIO_C", "PRODUTO_FICTICIO_D", "PRODUTO_FICTICIO_E"])
COLUMNS = {
    "clientes": ["cliente_id", "data_cadastro", "setor", "regiao"],
    "contratos": ["contrato_id", "cliente_id", "data_inicio", "produto_id", "valor_mensal", "gratuito"],
    "nps": ["pesquisa_id", "cliente_id", "data_pesquisa", "nota_0_10"],
    "telemetria": ["cliente_id", "mes_referencia", "qtd_eventos_uso"],
    "futuro_reservado": ["cliente_id", "janela_fim", "renovou_90d", "adotou_produto_90d", "queda_uso_90d"],
    "verdade_simulada_reservada": ["cliente_id", "perfil_latente", "outlier_sintetico"],
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def write_chunk(outdir: Path, table: str, frame: pd.DataFrame, is_first: bool):
    # Gzip deterministic metadata: no timestamp and no original file name.
    path = outdir / f"{table}.csv.gz"
    with path.open("wb" if is_first else "ab") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0, compresslevel=6) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, header=is_first, float_format="%.5f")


def assemble_chunk(start: int, n: int, *, seed: int, scenario: str):
    # Seeded per chunk: identical files for same seed, scenario, chunk size.
    rng = np.random.default_rng(seed)
    ids = np.array([f"S{i:09d}" for i in range(start + 1, start + n + 1)])
    base = rng.normal(0, 1, n)
    if scenario == "continuum":
        latent = np.full(n, -1, dtype=np.int8)
        impact_size = 0.0
        impact_sat = 0.0
    else:
        latent = rng.choice(3, n, p=[0.3, 0.45, 0.25]).astype(np.int8)
        impact_size = 1.1 if scenario == "distinct" else 0.35
        impact_sat = 2.0 if scenario == "distinct" else 0.65
    outlier = rng.uniform(size=n) < 0.005
    size_signal = base + (latent - 1) * impact_size * (latent >= 0)
    satisfaction = np.clip(6.5 - 0.7 * base + (latent - 1) * impact_sat * (latent >= 0) + rng.normal(0, 1.35, n), 0, 10)
    n_contracts = np.clip(rng.poisson(np.clip(np.exp(0.82 + 0.30 * size_signal), 0.4, 12)), 1, 14).astype(int)
    n_contracts[outlier] += 1
    registered = rng.integers(0, 2670, n)  # from 2019-01-01, before 2026-09-30
    registered_date = pd.Timestamp("2019-01-01") + pd.to_timedelta(registered, unit="D")
    sectors = rng.choice(SECTORS, n)
    regions = rng.choice(REGIONS, n)
    clients = pd.DataFrame({"cliente_id": ids, "data_cadastro": registered_date.strftime("%Y-%m-%d"), "setor": sectors, "regiao": regions})

    # Contract events only within the historical observation window.
    indices = np.repeat(np.arange(n), n_contracts)
    contract_ids = np.array([f"C{i:012d}" for i in range(start * 20 + 1, start * 20 + 1 + len(indices))])
    cutoff_days = (pd.Timestamp(CUTOFF) - pd.Timestamp("2019-01-01")).days
    begin_offset = registered[indices] + (rng.random(len(indices)) * (cutoff_days - registered[indices] + 1)).astype(int)
    begin_date = pd.Timestamp("2019-01-01") + pd.to_timedelta(begin_offset, unit="D")
    free_prob = sigmoid(-1.25 - 0.18 * size_signal[indices] + 0.1 * (satisfaction[indices] - 6))
    free = rng.random(len(indices)) < free_prob
    value = rng.lognormal(mean=5.2 + 0.54 * size_signal[indices], sigma=0.7)
    value *= np.where(outlier[indices], 24, 1)
    value = np.where(free, 0, value).round(2)
    contracts = pd.DataFrame({"contrato_id": contract_ids, "cliente_id": ids[indices], "data_inicio": begin_date.strftime("%Y-%m-%d"), "produto_id": rng.choice(PRODUCTS, len(indices)), "valor_mensal": value, "gratuito": free.astype(int)})

    surveyed = rng.random(n) < sigmoid(0.9 + 0.1 * size_signal)
    survey_ix = np.flatnonzero(surveyed)
    # Survey dates cannot precede customer registration.
    survey_start = max(0, (pd.Timestamp("2026-04-01") - pd.Timestamp("2019-01-01")).days)
    survey_lower = np.maximum(registered[survey_ix], survey_start)
    survey_offset = survey_lower + (rng.random(len(survey_ix)) * (cutoff_days - survey_lower + 1)).astype(int)
    survey_date = pd.Timestamp("2019-01-01") + pd.to_timedelta(survey_offset, unit="D")
    nps = pd.DataFrame({"pesquisa_id": [f"N{start+i+1:09d}" for i in survey_ix], "cliente_id": ids[survey_ix], "data_pesquisa": survey_date.strftime("%Y-%m-%d"), "nota_0_10": np.rint(satisfaction[survey_ix]).astype(int)})

    covered = rng.random(n) < sigmoid(0.25 + 0.32 * size_signal)
    covered_ix = np.flatnonzero(covered)
    telemetry_ix = np.repeat(covered_ix, 3)
    usage_lam = np.clip(np.exp(1.2 + 0.42 * size_signal[telemetry_ix]), 0.2, 100)
    telemetry = pd.DataFrame({"cliente_id": ids[telemetry_ix], "mes_referencia": np.tile(["2026-07-01", "2026-08-01", "2026-09-01"], len(covered_ix)), "qtd_eventos_uso": rng.poisson(usage_lam)})

    adoption_prob = sigmoid(-1.35 + 0.32 * size_signal + 0.23 * (satisfaction - 6) - 0.08 * n_contracts)
    renew_prob = sigmoid(0.55 + 0.33 * (satisfaction - 6) - 0.1 * base)
    drop_prob = sigmoid(-1.3 - 0.42 * (satisfaction - 6) + 0.13 * base)
    future = pd.DataFrame({"cliente_id": ids, "janela_fim": "2026-12-31", "renovou_90d": (rng.random(n) < renew_prob).astype(int), "adotou_produto_90d": (rng.random(n) < adoption_prob).astype(int), "queda_uso_90d": (rng.random(n) < drop_prob).astype(int)})
    truth = pd.DataFrame({"cliente_id": ids, "perfil_latente": latent, "outlier_sintetico": outlier.astype(int)})
    return {"clientes": clients, "contratos": contracts, "nps": nps, "telemetria": telemetry, "futuro_reservado": future, "verdade_simulada_reservada": truth}


def generate(outdir: Path, *, scenario: str, n_clients: int, seed: int, chunk_size: int = 10000):
    if scenario not in SCENARIOS: raise ValueError(f"Invalid scenario: {scenario}")
    if n_clients < 1 or chunk_size < 1: raise ValueError("n_clients/chunk_size must be positive")
    if outdir.exists() and any(outdir.iterdir()): raise FileExistsError(f"Refusing to overwrite nonempty directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in COLUMNS}
    for offset in range(0, n_clients, chunk_size):
        chunk_idx = offset // chunk_size
        # Independent deterministic seed per chunk; no hidden global RNG.
        chunk_seed = int(np.random.SeedSequence([seed, SCENARIOS.index(scenario), chunk_idx]).generate_state(1)[0])
        frames = assemble_chunk(offset, min(chunk_size, n_clients - offset), seed=chunk_seed, scenario=scenario)
        for name, frame in frames.items():
            write_chunk(outdir, name, frame.loc[:, COLUMNS[name]], counts[name] == 0)
            counts[name] += len(frame)
    files = {}
    for name in COLUMNS:
        path = outdir / f"{name}.csv.gz"
        files[path.name] = {"rows": counts[name], "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    manifest = {"generator": "TOTVS 2.0 original synthetic B2B demonstration", "generator_version": VERSION, "scenario": scenario, "n_clients": n_clients, "seed": seed, "chunk_size": chunk_size, "cutoff": CUTOFF, "forward_window_end": "2026-12-31", "disclaimer": "Fictitious records; not sampled, calibrated, inferred, or derived from TOTVS proprietary data. Not representative of real companies.", "files": files}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--clients", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(args.out, scenario=args.scenario, n_clients=args.clients, seed=args.seed, chunk_size=args.chunk_size)
    print(json.dumps({"out": str(args.out), "scenario": args.scenario, "rows": {k: v['rows'] for k,v in manifest['files'].items()}}, ensure_ascii=False))


if __name__ == "__main__": main()
