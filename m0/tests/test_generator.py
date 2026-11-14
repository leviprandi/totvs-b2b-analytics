from pathlib import Path
import sys
import tempfile
import json

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_synthetic import generate, CUTOFF


def test_scenarios_reproducibility_and_contracts():
    for scen in ("distinct", "overlap", "continuum"):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t)/"a", Path(t)/"b"
            ma = generate(a, scenario=scen, n_clients=205, seed=2027, chunk_size=100)
            mb = generate(b, scenario=scen, n_clients=205, seed=2027, chunk_size=100)
            assert ma == mb
            clients = pd.read_csv(a/"clientes.csv.gz")
            contracts = pd.read_csv(a/"contratos.csv.gz")
            nps = pd.read_csv(a/"nps.csv.gz")
            tel = pd.read_csv(a/"telemetria.csv.gz")
            future = pd.read_csv(a/"futuro_reservado.csv.gz")
            truth = pd.read_csv(a/"verdade_simulada_reservada.csv.gz")
            assert clients.cliente_id.is_unique
            assert future.cliente_id.is_unique
            assert truth.cliente_id.is_unique
            assert len(clients) == len(future) == len(truth) == 205
            assert contracts.contrato_id.is_unique
            assert contracts.cliente_id.isin(clients.cliente_id).all()
            assert nps.cliente_id.isin(clients.cliente_id).all()
            assert tel.cliente_id.isin(clients.cliente_id).all()
            assert contracts.data_inicio.max() <= CUTOFF
            assert nps.data_pesquisa.max() <= CUTOFF
            assert (nps.merge(clients, on='cliente_id').data_pesquisa >= nps.merge(clients, on='cliente_id').data_cadastro).all()
            assert tel.mes_referencia.max() <= CUTOFF
            assert future.janela_fim.min() > CUTOFF
            assert ((contracts.valor_mensal == 0) == (contracts.gratuito == 1)).all()
            if scen == 'continuum': assert (truth.perfil_latente == -1).all()
            else: assert truth.perfil_latente.between(0, 2).all()


def test_refuse_to_overwrite():
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'x'
        generate(p,scenario='distinct',n_clients=1,seed=1)
        try: generate(p,scenario='distinct',n_clients=1,seed=1)
        except FileExistsError: pass
        else: raise AssertionError('overwrite was not rejected')
