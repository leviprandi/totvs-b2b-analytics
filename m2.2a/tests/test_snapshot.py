import csv
import gzip
import hashlib
import io
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from snapshot import (
    FEATURES, KEY, OBSERVED, SCHEMAS, build_snapshot, load_manifest, split_snapshot,
    training_matrix, validate_reserved,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
DISTINCT = SAMPLES / "distinct_5k"


def rewrite_table(dataset, table, transform):
    name = f"{table}.csv.gz"
    path = dataset / name
    original = list(csv.reader(io.StringIO(gzip.decompress(path.read_bytes()).decode())))
    rows = transform(original)
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(rows)
    payload = gzip.compress(buffer.getvalue().encode(), mtime=0)
    path.write_bytes(payload)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = {
        "rows": len(rows) - 1, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
    }
    manifest_path.write_text(json.dumps(manifest))


@pytest.fixture
def dataset(tmp_path):
    return shutil.copytree(DISTINCT, tmp_path / "dataset")


@pytest.mark.parametrize("name,count", [
    ("distinct_5k", 5000), ("overlap_5k", 5000), ("continuum_5k", 5000),
    ("overlap_100k_stress", 100000),
])
def test_handoff_datasets(name, count):
    directory = SAMPLES / name
    dataset_id = validate_reserved(directory)
    snapshot = build_snapshot(directory, expected_dataset_id=dataset_id)
    assert len(snapshot) == count
    assert len({tuple(row[field] for field in KEY) for row in snapshot}) == count
    assert any(row["telemetria_observada"] == 0 and row["qtd_eventos_uso"] is None
               for row in snapshot)
    assert any(row["telemetria_observada"] == 1 and row["qtd_eventos_uso"] == 0
               for row in snapshot)


def test_order_invariance_and_disjoint_split(dataset):
    expected = build_snapshot(DISTINCT)
    for table in OBSERVED:
        def shuffle(rows):
            body = rows[1:]
            random.Random(2027).shuffle(body)
            return rows[:1] + body

        rewrite_table(dataset, table, shuffle)
    actual = build_snapshot(dataset)
    assert actual == expected
    splits = split_snapshot(actual, seed=101)
    assert splits == split_snapshot(list(reversed(actual)), seed=101)
    assert splits != split_snapshot(actual, seed=102)
    assert len({row["cliente_id"] for row in splits}) == 5000
    assert Counter(row["split"] for row in splits) == {"train": 3000, "reference": 1000, "test": 1000}


@pytest.mark.parametrize("table,column,value,error", [
    ("contratos", 0, "C-INVALID", "invalid contrato_id"),
    ("contratos", 1, "S999999999", "unknown cliente_id"),
    ("contratos", 1, "invalid", "invalid cliente_id"),
    ("contratos", 2, "2018-12-31", "outside observation window"),
    ("contratos", 4, "NaN", "invalid valor_mensal"),
    ("nps", 2, "2026-10-01", "outside observation window"),
    ("nps", 3, "11", "invalid pesquisa_id or score"),
    ("telemetria", 1, "2026-08-02", "month must start on day 1"),
    ("telemetria", 2, "-1", "expected a non-negative integer"),
])
def test_reject_invalid_observations(dataset, table, column, value, error):
    def alter(rows):
        rows[1][column] = value
        return rows

    rewrite_table(dataset, table, alter)
    with pytest.raises(ValueError, match=error):
        build_snapshot(dataset)


@pytest.mark.parametrize("table", OBSERVED)
def test_reject_duplicate_keys(dataset, table):
    rewrite_table(dataset, table, lambda rows: rows + rows[1:2])
    with pytest.raises(ValueError, match="duplicate key"):
        build_snapshot(dataset)


def test_wrong_dataset_and_file_mixing(dataset):
    _, other = load_manifest(SAMPLES / "overlap_5k")
    with pytest.raises(ValueError, match="expected dataset_id"):
        build_snapshot(dataset, expected_dataset_id=other)
    shutil.copyfile(SAMPLES / "overlap_5k" / "contratos.csv.gz", dataset / "contratos.csv.gz")
    with pytest.raises(ValueError, match="content does not match dataset manifest"):
        build_snapshot(dataset)


def test_reserved_columns_cannot_reach_training(dataset):
    original = build_snapshot(dataset)
    for table in set(SCHEMAS) - set(OBSERVED):
        (dataset / f"{table}.csv.gz").unlink()
    snapshot = build_snapshot(dataset)
    assert snapshot == original
    assert all(tuple(row) == FEATURES for row in training_matrix(snapshot))
    with pytest.raises(ValueError, match="unexpected or missing columns"):
        training_matrix([{**snapshot[0], "perfil_latente": 0}])
    with pytest.raises(ValueError, match="duplicate key"):
        split_snapshot(snapshot[:1] * 3, seed=101)

    rewrite_table(dataset, "clientes", lambda rows: [
        row + [("renovou_90d" if i == 0 else "0")] for i, row in enumerate(rows)
    ])
    with pytest.raises(ValueError, match="unexpected columns"):
        build_snapshot(dataset)


def test_cli_exports_only_training_features(tmp_path):
    out = tmp_path / "export"
    subprocess.run(
        [sys.executable, str(ROOT / "snapshot.py"), "--dataset", str(DISTINCT), "--out", str(out)],
        check=True,
    )
    with (out / "features.csv").open(newline="") as stream:
        records = list(csv.reader(stream))
    assert tuple(records[0]) == FEATURES
    assert len(records) == 3001
    assert json.loads((out / "snapshot.json").read_text())["train_rows"] == 3000


@pytest.mark.parametrize("name", ["distinct_5k", "overlap_5k", "continuum_5k", "overlap_100k_stress"])
def test_regenerate_handoff_manifest(tmp_path, name):
    from generate_synthetic import generate

    source = SAMPLES / name
    expected = json.loads((source / "manifest.json").read_text())
    output = tmp_path / name
    actual = generate(
        output, scenario=expected["scenario"], n_clients=expected["n_clients"],
        seed=expected["seed"], chunk_size=expected["chunk_size"],
    )
    assert {k: v for k, v in actual.items() if k != "files"} == {
        k: v for k, v in expected.items() if k != "files"
    }
    for name, metadata in expected["files"].items():
        assert actual["files"][name]["rows"] == metadata["rows"]
        # Compression can differ across zlib versions, even for identical CSVs.
        with gzip.open(source / name, "rb") as original, gzip.open(output / name, "rb") as regenerated:
            assert hashlib.file_digest(original, "sha256").digest() == hashlib.file_digest(
                regenerated, "sha256"
            ).digest()


def test_aggregations_at_cutoff(dataset):
    from decimal import Decimal

    client = "S000000001"
    expected = {
        "contratos": [
            ["C000000000001", client, "2026-09-29", "PRODUTO_A", "0.00000", "1"],
            ["C000000000002", client, "2026-09-30", "PRODUTO_A", "100.12000", "0"],
            ["C000000000003", client, "2026-09-30", "PRODUTO_B", "12.35000", "0"],
        ],
        "nps": [
            ["N000000001", client, "2026-09-29", "4"],
            ["N000000002", client, "2026-09-30", "9"],
        ],
        "telemetria": [[client, "2026-09-01", "0"]],
    }
    for table, records in expected.items():
        rewrite_table(dataset, table, lambda rows: rows[:1] + records)
    row = build_snapshot(dataset)[0]
    assert row["cliente_id"] == client
    assert row["data_corte"] == "2026-09-30"
    assert row["valor_mensal_total"] == Decimal("112.47")
    assert (row["qtd_contratos"], row["qtd_produtos"]) == (3, 2)
    assert row["proporcao_gratuito"] == pytest.approx(1 / 3)
    assert row["nps_medio"] == 6.5
    assert (row["telemetria_observada"], row["qtd_eventos_uso"]) == (1, 0)


def test_reject_mixed_snapshots():
    distinct = build_snapshot(DISTINCT)
    overlap = build_snapshot(SAMPLES / "overlap_5k")
    with pytest.raises(ValueError, match="exactly one dataset"):
        split_snapshot(distinct[:2] + overlap[:1], seed=101)
