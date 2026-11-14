# M2.2B — protocolo v1.0

Hipóteses: C1 (uso), C3 (sem telemetria) e C4 (valor e uso com cap que
preserva zero) oferecem informação futura além de regras observáveis?
Uma divisão estável em continuum pode ser apenas uma discretização.

- Três datasets de 5.000 clientes. Split M0 por dataset (semente 101);
  3.000 treino, 1.000 referência, 1.000 teste. Reutilizar C1/C3/C4 de
  controls.py, k=2, seed 11, parâmetros inalterados (k-means++, n_init=10,
  max_iter=300, tol=1e-4). Ajustar no treino de cada cenário. Reaprender em
  cinco subamostras de 80% sem reposição, seeds 1011–1015 x 11/29. Filtro
  M2.2A inalterado. Total planejado 99, teto 100 e 600 s entre ajustes.
  Persistir por ajuste. Se incompleto, encerrar sem abrir teste.
- Regras: valor total == 0; uso observado <=7 e >=18 (cortes exploratórios
  históricos do treino original, fornecidos na autorização e congelados,
  sem reestimativa; ausentes fora dos dois grupos); cobertura observada;
  valor >= mediana do treino; cruzamento valor/cobertura (até quatro
  grupos). Para controle de prioridade, comparar ambos os grupos de cada
  modelo com o mesmo N nos rankings baixo/alto valor e baixo/alto uso
  observado. Desempate por chave ordenada. Uso sem N observações fica
  indisponível. Comparações de tamanho equivalente são descritivas.
- Desfechos renovou_90d, adotou_produto_90d e queda_uso_90d (nomes legados,
  trimestre sintético de 92 dias). Reportar N, eventos, taxa, contraste
  grupo vs complemento em p.p., razão de riscos quando definida e V de
  Cramér para associação global. IC descritivo de 95% por bootstrap
  estratificado de clientes dentro/fora do grupo, 2.000 repetições, seed
  4021. Para contrastes de tamanho equivalente, parear por cliente e
  reamostrar os mesmos índices para ambas as regras (2.000).
- Informação incremental: na referência, previsão OOF em cinco folds
  fixados por chave (seed 4022), com suavização Beta(1,1) nas taxas por
  célula. Baseline conjunto cruza as seis regras; o concorrente acrescenta
  o rótulo do modelo. Células novas recuam para a taxa global do treino do
  fold no baseline e para o baseline no concorrente. Comparar Brier
  pareado (baseline menos concorrente) por desfecho. Uma regra isolada é
  comparada com a taxa global. IC por bootstrap pareado das perdas OOF
  fixas (10.000 repetições, seed 4023). Por cenário, Bonferroni para
  27 comparações (3 modelos + 6 regras, 3 desfechos). O IC é condicional
  às previsões OOF e não incorpora o reajuste do estimador.
- Critério pré-fixado: avanço de modelo exige ajuste inicial elegível,
  estabilidade M2.2A, e ganho Brier >=0,002 com limite inferior ajustado
  acima de zero em pelo menos um desfecho de distinct e mesmo desfecho
  em overlap, com estabilidade também em overlap. Se mais de um par
  passar, ordenar pelo maior mínimo do ganho nos dois cenários e
  desempatar lexicalmente. Na ausência de modelo, aplicar o mesmo critério
  de ganho para regra isolada. Se nenhum passar, decisão de seleção
  'nenhum', com ganho incremental inconclusivo quando a incerteza
  incluir zero. Continuum é diagnóstico e não seleciona candidatos.
- Abrir teste uma vez, somente após registrar seleção, listas de métodos,
  hashes de código/protocolo/entradas e referências. Sem refit da
  segmentação ou dos limites: os modelos treinados em treino, seed 11,
  são congelados. Na confirmação, ajustar apenas as tabelas de risco na
  referência inteira. Avaliar os métodos congelados uma vez nos três
  testes; resultados exploratórios para os métodos não selecionados.
  Para o par selecionado, confirmar apenas se nos dois cenários de
  seleção o ganho >=0,002 e os limites inferiores ajustados >0. Sem
  seleção, usar a confirmação para reportar os comparadores descritivos,
  sem promover candidato. Falha após marcar a abertura consome a reserva.
- Não usar verdade latente, não estimar causalidade/ROI. Hash de código,
  versão, dependências e data registrados antes de ler desfechos.

## Registro técnico anterior a desfechos

A primeira execução de ajuste persistiu 22 artefatos. O primeiro ajuste
C4 terminou, mas a verificação da transformação congelada esperava a chave
opcional log1p. O ajuste foi perdido antes da persistência. A recuperação
copia os 22 ajustes concluídos e debita a tentativa perdida do teto.
Somente a leitura da chave opcional e a recuperação foram ajustadas.
Hipóteses, populações, métricas e critérios acima permanecem como registrados.
