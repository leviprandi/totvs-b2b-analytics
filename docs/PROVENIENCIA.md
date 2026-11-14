# Proveniência da edição pública

A extensão M0–M2.2B usa dados sintéticos produzidos pelo gerador incluído.
O desafio acadêmico original foi realizado em equipe e está descrito no README
como contexto. Esta edição omite bases, código, slides, vídeos, dashboards,
logos e credenciais da entrega acadêmica ou do parceiro. Eventual inclusão desses
materiais requer revisão de direitos e das condições de compartilhamento.

## Evidência experimental

- Código por etapa, requisitos e protocolo foram copiados byte a byte.
- Os manifestos de entrada preservam a descrição completa de cada fixture.
  A árvore pública contém somente as quatro tabelas observadas. Os hashes dos
  arquivos reservados permanecem como registro no manifesto. A base de volume
  tem apenas o manifesto em `m0/samples/overlap_100k_stress/`.
- O controle M2.2A depende das atribuições e do manifesto históricos de
  `P1_clip` em M2.1. Esses arquivos e as métricas do ajuste foram preservados.
- Mantivemos os relatórios agregados de referência e de teste do M2.2B e os
  nove ajustes iniciais que o manifesto final referencia por hash. As listas de
  ajustes e a recuperação descrevem a execução integral do acervo. A árvore
  pública não inclui as atribuições por cliente das reamostragens, a execução
  parcial recuperada ou os CSVs brutos de desfechos e verdade latente.
- `manifest.public.json` do M2.2A e os dois marcadores de abertura do M2.2B
  são cópias derivadas identificadas. Cada uma informa o SHA-256 do original
  e o campo de caminho substituído. Os marcadores preservam o horário e o hash
  da decisão congelada. O marcador de raiz continua bloqueando a confirmação.
- O [relatório público](reports/challenge_totvs_2_relatorio_tecnico_publico.pdf)
  deriva do relatório ampliado, com contexto acadêmico e referências de
  evidência ajustadas à árvore pública. A edição preserva os resultados e a
  cronologia do protocolo. O PDF e o DOCX originais ficam no acervo.

O [inventário SHA-256](ARQUIVOS_SHA256.txt) enumera os arquivos desta edição,
com exceção do próprio inventário. As confirmações históricas são evidência
fechada. A documentação de reprodução cobre preparação e ajuste com dados
observados. Para interpretação descritiva, use os relatórios agregados.

## Fontes do relatório público

- `challenge_totvs_2_relatorio_tecnico_ampliado.docx` no acervo. SHA-256 `c6248d1ae627afb816fda94ebdc980d9de16e27c253e3ecf8ab58d66d41a381c`.
- `challenge_totvs_2_relatorio_tecnico_ampliado.pdf` no acervo. SHA-256 `623f72929ed5ab2c8df440aaf8547995562bf278794ef97ea692310b032a1b76`.
