# M2.2B — Validação sintética

O [protocolo](PROTOCOLO.md) preserva o desenho e as regras da execução histórica.
A [decisão registrada](results/validation-v1-recovered/confirmation.json) é
`no_method`. A [reserva está consumida](TESTE_ABERTO.json); o código de
confirmação é mantido para inspeção. Os desfechos já agregados estão nos JSONs
de referência e teste da execução consolidada.

A única fase de execução documentada para esta versão é
`validation.py fit --out <diretório-novo>`. Ela usa atributos observados e
não carrega os desfechos reservados. Veja os comandos, escopo dos fixtures e
limites de reprodução no [README principal](../README.md). A suíte integral
original depende dos fixtures reservados e de volume mantidos no acervo.
