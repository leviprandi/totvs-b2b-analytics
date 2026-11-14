"""Validate a synthetic dataset and export its observed snapshot and split."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import re
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path


SCHEMAS = {
    "clientes": "cliente_id data_cadastro setor regiao",
    "contratos": "contrato_id cliente_id data_inicio produto_id valor_mensal gratuito",
    "nps": "pesquisa_id cliente_id data_pesquisa nota_0_10",
    "telemetria": "cliente_id mes_referencia qtd_eventos_uso",
    "futuro_reservado": "cliente_id janela_fim renovou_90d adotou_produto_90d queda_uso_90d",
    "verdade_simulada_reservada": "cliente_id perfil_latente outlier_sintetico",
}
OBSERVED = ("clientes", "contratos", "nps", "telemetria")
PRIMARY_KEYS = {
    "contratos": ("contrato_id",),
    "nps": ("pesquisa_id",),
    "telemetria": ("cliente_id", "mes_referencia"),
}
IDENTITY_FIELDS = (
    "generator", "generator_version", "scenario", "n_clients", "seed",
    "chunk_size", "cutoff", "forward_window_end",
)
KEY = ("dataset_id", "cliente_id", "data_corte")
FEATURES = (
    "valor_mensal_total", "qtd_contratos", "qtd_produtos", "dias_relacionamento",
    "proporcao_gratuito", "nps_medio", "telemetria_observada", "qtd_eventos_uso",
)
SNAPSHOT_COLUMNS = KEY + FEATURES + ("setor", "regiao")


def iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("expected a date in YYYY-MM-DD format")
    return date.fromisoformat(value)


def unsigned(value):
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError("expected a non-negative integer")
    return int(value)


def load_manifest(directory, *, expected_dataset_id=None):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    try:
        identity = {field: manifest[field] for field in IDENTITY_FIELDS}
        cutoff = iso_date(identity["cutoff"])
        if iso_date(identity["forward_window_end"]) <= cutoff:
            raise ValueError("future window must end after cutoff")
        if identity["scenario"] not in {"distinct", "overlap", "continuum"}:
            raise ValueError("unknown scenario")
        for field, minimum in (("n_clients", 1), ("seed", 0), ("chunk_size", 1)):
            if type(identity[field]) is not int or identity[field] < minimum:
                raise ValueError(f"invalid manifest field {field}")
        for field in ("generator", "generator_version"):
            if not isinstance(identity[field], str) or not identity[field].strip():
                raise ValueError(f"invalid manifest field {field}")
        if set(manifest["files"]) != {f"{name}.csv.gz" for name in SCHEMAS}:
            raise ValueError("unexpected manifest files")
        for file, metadata in manifest["files"].items():
            if (
                type(metadata["rows"]) is not int or metadata["rows"] < 0
                or type(metadata["bytes"]) is not int or metadata["bytes"] < 1
                or not re.fullmatch(r"[a-f0-9]{64}", metadata["sha256"])
            ):
                raise ValueError(f"invalid manifest metadata for {file}")
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid manifest schema") from exc
    dataset_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise ValueError("manifest does not match expected dataset_id")
    return manifest, dataset_id


def read_table(directory, manifest, table):
    file = f"{table}.csv.gz"
    metadata = manifest["files"][file]
    payload = (directory / file).read_bytes()
    if len(payload) != metadata["bytes"] or hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError(f"{table}: content does not match dataset manifest")
    columns = SCHEMAS[table].split()
    primary = PRIMARY_KEYS.get(table, ("cliente_id",))
    seen = set()
    with gzip.open(io.BytesIO(payload), "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != columns:
            raise ValueError(f"{table}: unexpected columns or column order")
        for line, fields in enumerate(reader, 2):
            if len(fields) != len(columns) or any(not value.strip() for value in fields):
                raise ValueError(f"{table}:{line}: invalid field count or empty field")
            row = dict(zip(columns, fields))
            if not re.fullmatch(r"S[0-9]{9}", row["cliente_id"]):
                raise ValueError(f"{table}:{line}: invalid cliente_id")
            key = tuple(row[column] for column in primary)
            if key in seen:
                raise ValueError(f"{table}:{line}: duplicate key")
            seen.add(key)
            yield row
    if len(seen) != metadata["rows"]:
        raise ValueError(f"{table}: row count does not match dataset manifest")


def load_clients(directory, manifest):
    clients = {}
    cutoff = iso_date(manifest["cutoff"])
    for row in read_table(directory, manifest, "clientes"):
        registration = iso_date(row["data_cadastro"])
        if registration > cutoff:
            raise ValueError("clientes: registration after cutoff")
        clients[row["cliente_id"]] = (registration, row["setor"], row["regiao"])
    if len(clients) != manifest["n_clients"]:
        raise ValueError("clientes: population does not match manifest")
    return clients


def build_snapshot(directory, *, expected_dataset_id=None):
    """Open only observed tables, with membership tied to a single manifest."""
    manifest, dataset_id = load_manifest(directory, expected_dataset_id=expected_dataset_id)
    clients = load_clients(directory, manifest)
    cutoff = iso_date(manifest["cutoff"])
    contracts = defaultdict(list)
    surveys = defaultdict(list)
    telemetry = {}
    date_fields = {"contratos": "data_inicio", "nps": "data_pesquisa", "telemetria": "mes_referencia"}
    for table in OBSERVED[1:]:
        for row in read_table(directory, manifest, table):
            client = row["cliente_id"]
            if client not in clients:
                raise ValueError(f"{table}: unknown cliente_id")
            event = iso_date(row[date_fields[table]])
            if not clients[client][0] <= event <= cutoff:
                raise ValueError(f"{table}: event outside observation window")
            if table == "contratos":
                if not re.fullmatch(r"C[0-9]{12}", row["contrato_id"]):
                    raise ValueError("contratos: invalid contrato_id")
                amount = row["valor_mensal"]
                if not re.fullmatch(r"[0-9]+(?:\.[0-9]{1,5})?", amount):
                    raise ValueError("contratos: invalid valor_mensal")
                free = row["gratuito"]
                value = Decimal(amount)
                if free not in {"0", "1"} or (free == "1" and value != 0):
                    raise ValueError("contratos: invalid gratuito or value for free contract")
                contracts[client].append((value, row["produto_id"], free == "1"))
            elif table == "nps":
                score = unsigned(row["nota_0_10"])
                if score > 10 or not re.fullmatch(r"N[0-9]{9}", row["pesquisa_id"]):
                    raise ValueError("nps: invalid pesquisa_id or score")
                surveys[client].append(score)
            else:
                if event.day != 1:
                    raise ValueError("telemetria: month must start on day 1")
                telemetry[client] = telemetry.get(client, 0) + unsigned(row["qtd_eventos_uso"])

    rows = []
    for client, (registration, sector, region) in sorted(clients.items()):
        entries = contracts.get(client, ())
        scores = surveys.get(client, ())
        count = len(entries)
        rows.append(dict(zip(SNAPSHOT_COLUMNS, (
            dataset_id, client, cutoff.isoformat(),
            sum((entry[0] for entry in entries), Decimal("0.00")),
            count, len({entry[1] for entry in entries}), (cutoff - registration).days,
            sum(entry[2] for entry in entries) / count if count else None,
            sum(scores) / len(scores) if scores else None,
            int(client in telemetry), telemetry.get(client), sector, region,
        ))))
    return rows


def validate_reserved(directory, *, expected_dataset_id=None):
    """Validate evaluation artifacts separately from snapshot and training."""
    manifest, dataset_id = load_manifest(directory, expected_dataset_id=expected_dataset_id)
    clients = load_clients(directory, manifest)
    for table in ("futuro_reservado", "verdade_simulada_reservada"):
        ids = set()
        for row in read_table(directory, manifest, table):
            ids.add(row["cliente_id"])
            if table == "futuro_reservado":
                if row["janela_fim"] != manifest["forward_window_end"]:
                    raise ValueError("futuro_reservado: unexpected window end")
                binary = SCHEMAS[table].split()[2:]
            else:
                valid_profiles = {"-1"} if manifest["scenario"] == "continuum" else {"0", "1", "2"}
                if row["perfil_latente"] not in valid_profiles:
                    raise ValueError("verdade_simulada_reservada: unexpected latent label")
                binary = ("outlier_sintetico",)
            if any(row[field] not in {"0", "1"} for field in binary):
                raise ValueError(f"{table}: invalid binary field")
        if ids != clients.keys():
            raise ValueError(f"{table}: client keys differ from observed population")
    return dataset_id


def snapshot_keys(rows):
    if any(row.keys() != set(SNAPSHOT_COLUMNS) for row in rows):
        raise ValueError("snapshot: unexpected or missing columns")
    keys = [tuple(row[column] for column in KEY) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("snapshot: duplicate key")
    if len({(dataset, cutoff) for dataset, _, cutoff in keys}) != 1:
        raise ValueError("snapshot: expected exactly one dataset and cutoff")
    return keys


def training_matrix(rows):
    """Return only allowlisted features, in snapshot row order."""
    snapshot_keys(rows)
    return [{field: row[field] for field in FEATURES} for row in rows]


def split_snapshot(rows, *, seed):
    keys = snapshot_keys(rows)
    if len(keys) < 3:
        raise ValueError("split requires at least three clients")
    ranked = sorted(keys, key=lambda key: (
        hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).digest(), key,
    ))
    train_end = len(keys) * 60 // 100
    reference_end = train_end + max(1, len(keys) * 20 // 100)
    partitions = {
        key: "train" if i < train_end else "reference" if i < reference_end else "test"
        for i, key in enumerate(ranked)
    }
    return [dict(zip((*KEY, "split"), (*key, partitions[key]))) for key in sorted(keys)]


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--expected-dataset-id")
    args = parser.parse_args()
    try:
        rows = build_snapshot(args.dataset, expected_dataset_id=args.expected_dataset_id)
        splits = split_snapshot(rows, seed=args.seed)
        train_ids = {row["cliente_id"] for row in splits if row["split"] == "train"}
        matrix = training_matrix([row for row in rows if row["cliente_id"] in train_ids])
        args.out.mkdir(parents=True)
        write_csv(args.out / "snapshot.csv", SNAPSHOT_COLUMNS, rows)
        write_csv(args.out / "split.csv", (*KEY, "split"), splits)
        write_csv(args.out / "features.csv", FEATURES, matrix)
        metadata = {"dataset_id": rows[0]["dataset_id"], "data_corte": rows[0]["data_corte"],
                    "seed": args.seed, "rows": len(rows), "train_rows": len(matrix),
                    "source_manifest_sha256": hashlib.sha256(
                        (args.dataset / "manifest.json").read_bytes()).hexdigest()}
        (args.out / "snapshot.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, csv.Error, EOFError) as exc:
        parser.exit(2, f"M0: {exc}\n")


if __name__ == "__main__":
    main()
