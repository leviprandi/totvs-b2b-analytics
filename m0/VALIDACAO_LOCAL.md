# Registro de validação de material sintético

- Executor: ambiente temporário isolado do ChatGPT, não o Mac do usuário.
- Fonte: gerador original desenvolvido nesta sessão; não utiliza arquivos ou estatísticas proprietárias TOTVS.
- Testes: `python -m pytest -q tests/test_generator.py` -> **2 passed**.
- Sintaxe: `python -m compileall -q generate_synthetic.py tests/test_generator.py` -> concluído.
- Integridade: SHA-256 de todos os `.csv.gz` igual ao manifesto; contagens compatíveis; IDs de contrato únicos; nenhuma duplicata por cliente/mês na telemetria nos quatro conjuntos.
- Cenários: `distinct_5k`, `overlap_5k`, `continuum_5k`, `overlap_100k_stress`.
- Stress: 100.000 clientes, 249.901 contratos, 71.080 pesquisas NPS, 167.376 medições mensais de telemetria, 100.000 indicadores futuros e 100.000 perfis latentes reservados. **788.357 linhas** somando todas as tabelas. Sem treinamento ML.
- Separação temporal: contratos, pesquisas e telemetria até `2026-09-30`; outcomes em `2026-12-31`; nenhuma pesquisa antes do cadastro.
- Ausência ≠ zero: telemetria ausente por cliente sem cobertura; observações disponíveis também contêm zero válido.
- Limitação: os testes cobrem o contrato básico; não certificam adequação estatística nem desempenho/validade comercial. A arquitetura do gerador deve ser auditada independentemente pelo Codex.
