# Contrato de dados sintéticos TOTVS 2.0 — proposta v0.1

**Status:** material de apoio gerado fora do Mac; não aprovado como contrato definitivo nem integrado pelo Codex. Nenhuma estatística, cliente, produto ou estrutura restrita da TOTVS foi copiada ou utilizada na calibração. Exemplo fictício e não representativo.

- Unidade operacional: `cliente_id` único em `clientes`, `futuro_reservado` e `verdade_simulada_reservada`.
- Janela de observação: até `2026-09-30`; janela futura de avaliação: de `2026-10-01` a `2026-12-31`, inclusive (92 dias).
- `clientes`: cadastro e duas categorias (`setor`, `regiao`) fictícias; **sem rótulo latente**.
- `contratos`: zero ou mais contratos por cliente (na geração atual, pelo menos um); valor mensal em unidade monetária fictícia, não negativa; gratuito=1 implica valor=0.
- `nps`: pesquisa opcional, nota individual inteira 0–10 (não o indicador corporativo NPS agregado -100..100).
- `telemetria`: observações mensais de quantidade inteira de eventos; **ausência de linha não significa uso zero**. Uma linha com `qtd_eventos_uso=0` indica zero observado.
- `futuro_reservado`: indicadores binários (`renovou_90d`, `adotou_produto_90d`, `queda_uso_90d`), gerados a partir de mecanismos probabilísticos correlacionados aos dados observáveis, mas não dos clusters descobertos. Os sufixos `_90d` são nomes legados para os indicadores do trimestre documentado de 92 dias. A janela efetiva vem do corte e de `forward_window_end` do manifesto. Os eventos futuros são simulados como indicadores da janela, sem datas individuais. **Proibido usar em treino, pré-processamento ou seleção orientada a resultado do teste**.
- `verdade_simulada_reservada`: perfil latente (-1 para cenário contínuo) e flag de outlier sintético. **Proibido usar como feature**; pode ser inspecionado apenas na avaliação apropriada, com critérios aprovados.
- `manifest.json`: versão, configuração, semente, número de linhas, hashes SHA-256, tamanho e janela. O hash é do GZIP comprimido; a configuração completa (incluindo `chunk_size`) é parte da reprodutibilidade.

## Limitações relevantes
- Gerador demonstra contratos de esquema e escala de ingestão; **não é simulador validado do mercado B2B**.
- Os rótulos de perfil latente influenciam tamanho/satisfação no cenário distinto/sobreposto, mas não devem ser interpretados como perfis reais de cliente.
- Cenário contínuo não implica uma única distribuição esférica; algoritmos de clusterização ainda poderão impor cortes na distribuição.
- Há registros múltiplos de telemetria e contratos por cliente. **Sempre dividir treino/referência/teste por cliente**, após agregação/snapshot, não por linha de evento.
- Não há performance ou valor comercial comprovado pelo gerador sozinho.
- Arquivos em CSV.GZ portátil e gerados por chunks: para workloads de escala reais, considerar Parquet particionado e processamento por streaming/Polars/DuckDB após benchmark; a primeira fase não requer Spark.

## Consumo no M0

- `dataset_id` é o SHA-256 da configuração canônica do manifesto: gerador, versão, cenário, população, sementes, chunk e datas. A identidade é estável sob reordenação dos arquivos. Os hashes dos arquivos atestam o conteúdo da revisão do manifesto e detectam trocas de arquivos entre pacotes. Se o mesmo gerador produzir conteúdo distinto com a mesma configuração, a versão do gerador ou a configuração deve mudar antes de tratar as duas versões como datasets separados.
- `snapshot.csv` tem uma linha por `(dataset_id, cliente_id, data_corte)`. Contratos iniciados até o corte contribuem para soma mensal e contagens, sem pressupor vigência, pois a origem não tem data de encerramento. NPS é a média das respostas observadas. A telemetria soma os meses disponíveis, com cobertura 0 e uso vazio quando não há linha, e cobertura 1 e valor 0 quando o zero é observado. Sem contrato, a proporção de gratuidades é vazia.
- `split.csv` usa ranking de hash por dataset, cliente e semente, com proporções 60/20/20 e arredondamento para manter as três partições. `features.csv` contém apenas linhas de treino, na ordem das chaves de treino em `split.csv`, e oito atributos numéricos permitidos. Setor e região ficam no snapshot para caracterização. As tabelas reservadas têm validação separada.
