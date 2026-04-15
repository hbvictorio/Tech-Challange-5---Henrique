# Datathon Passos Magicos - Apresentacao Gerencial

## Slide 1: Capa
**Titulo:** Datathon Passos Magicos - Analise Preditiva de Risco de Defasagem
**Subtitulo:** Transformando dados em oportunidades para criancas e jovens
**Contexto:** Equipe Datathon | FIAP PosTech 2024
**Visual:** Fundo com gradiente azul escuro (#1B3A5C) para azul medio (#2E6B9E), logo da Passos Magicos centralizado

## Slide 2: A defasagem esta crescendo e exige atencao imediata
**Insight principal:** O IAN medio caiu de 7.43 (2020) para 6.42 (2022), uma queda de 13.6% em dois anos. A defasagem moderada cresceu de 49% para 67% dos alunos.
**Dados de suporte:** Grafico de barras mostrando IAN medio por ano (7.43, 6.90, 6.42) e grafico empilhado de distribuicao de defasagem (Severa, Moderada, Adequado) por ano.
**Conclusao:** A tendencia de queda no IAN indica que mais alunos estao ficando defasados em relacao ao nivel esperado, sinalizando necessidade de intervencao.

## Slide 3: O desempenho academico sofreu queda em 2021 mas mostra sinais de recuperacao
**Insight principal:** O IDA medio caiu de 6.32 (2020) para 5.43 (2021), mas recuperou parcialmente para 6.07 (2022). Individualmente, 52.5% dos alunos com dados longitudinais apresentaram queda no desempenho.
**Dados de suporte:** Box plot do IDA por ano e grafico de barras com tendencia individual (322 caindo, 187 melhorando, 104 estagnados).
**Conclusao:** A queda de 2021 pode estar associada ao impacto da pandemia. A recuperacao em 2022 e positiva, mas a maioria dos alunos ainda nao retornou ao nivel de 2020.

## Slide 4: Engajamento e o motor do desempenho e do ponto de virada
**Insight principal:** O IEG apresenta correlacao forte com IDA (0.57 a 0.68) e com IPV (0.35 a 0.73). Alunos mais engajados consistentemente apresentam melhor desempenho e maior probabilidade de atingir o ponto de virada.
**Dados de suporte:** Scatter plots IEG vs IDA e IEG vs IPV com cores por ano, mostrando relacao positiva clara.
**Conclusao:** Investir em estrategias de engajamento e a alavanca mais eficaz para melhorar resultados academicos e transformacao pessoal.

## Slide 5: Alunos superestimam seu desempenho - autoavaliacao tem vies positivo
**Insight principal:** A correlacao entre IAA e IDA e fraca (0.22-0.33), com vies positivo consistente de +2.05 a +2.73 pontos. Alunos sistematicamente avaliam seu desempenho acima do real.
**Dados de suporte:** Scatter IAA vs IDA com linha de igualdade e histograma do vies (IAA - IDA).
**Conclusao:** A autoavaliacao nao deve ser usada como unico indicador de progresso. Recomenda-se combinar IAA com metricas objetivas para uma visao mais precisa.

## Slide 6: IPS isolado nao prediz quedas, mas IPP e IAN medem dimensoes complementares
**Insight principal:** O IPS medio e similar entre alunos com e sem queda (6.74 vs 6.79). A correlacao IPP-IAN e muito fraca (-0.06 a 0.12), indicando que avaliacoes psicopedagogicas e adequacao ao nivel capturam aspectos diferentes do desenvolvimento.
**Dados de suporte:** Box plot comparativo IPS por grupo de queda e scatter IPP vs IAN.
**Conclusao:** Abordagem multidimensional e essencial. Nenhum indicador isolado e suficiente para prever risco.

## Slide 7: Os preditores do ponto de virada mudaram ao longo do tempo
**Insight principal:** Em 2020, o IPP (0.77) era o principal preditor do IPV. Em 2021-2022, IEG (0.73/0.60) e IDA (0.69/0.62) assumiram a lideranca. Isso sugere uma evolucao na dinamica do programa.
**Dados de suporte:** Grafico de barras agrupadas mostrando correlacao de cada indicador com IPV por ano.
**Conclusao:** A mudanca nos preditores pode refletir ajustes metodologicos ou o impacto da pandemia. O engajamento e o desempenho academico sao hoje os fatores mais criticos para o ponto de virada.

## Slide 8: IDA e IEG sao os pilares do INDE - modelo preditivo confirma
**Insight principal:** IDA (r=0.81) e IEG (r=0.80) sao os indicadores com maior correlacao com o INDE. O modelo preditivo alcancou AUC-ROC de 1.000 e F1-Score de 0.924, confirmando que esses indicadores sao altamente preditivos.
**Dados de suporte:** Matriz de correlacao (heatmap) e grafico de importancia de features do modelo.
**Conclusao:** Focar em aprendizagem e engajamento e a estrategia mais eficiente para elevar o desenvolvimento educacional global.

## Slide 9: Modelo preditivo identifica alunos em risco com alta precisao
**Insight principal:** O modelo de Regressao Logistica com 15 features alcancou Accuracy 98.2%, F1-Score 92.4% e AUC-ROC 100%. Alunos em risco apresentam IDA medio de 2.02 e IEG de 3.56, versus 6.43 e 7.97 dos alunos sem risco.
**Dados de suporte:** Curva ROC comparando 3 modelos, matriz de confusao e grafico de perfil medio (risco vs sem risco).
**Conclusao:** A ferramenta preditiva permite identificacao precoce de alunos em risco, possibilitando intervencao antes da queda efetiva no desempenho.

## Slide 10: O programa mantem impacto consistente, mas retencao e o padrao dominante
**Insight principal:** A distribuicao de pedras e relativamente estavel ao longo dos anos. De 457 alunos acompanhados de 2020 a 2021, 50.5% mantiveram a mesma pedra, 15.8% subiram e 33.3% desceram.
**Dados de suporte:** Grafico empilhado de distribuicao de pedras por ano e grafico de progressao de pedra.
**Conclusao:** O programa mantem sua base, mas ha espaco para aumentar a taxa de progressao. Estrategias focadas em engajamento e aprendizagem podem acelerar a mobilidade ascendente.

## Slide 11: Recomendacoes estrategicas baseadas nos dados
**Ponto 1:** Implementar sistema de alerta precoce usando o modelo preditivo para identificar alunos em risco antes da queda no desempenho.
**Ponto 2:** Priorizar estrategias de engajamento (IEG) como principal alavanca de melhoria, dado seu impacto comprovado no IDA e IPV.
**Ponto 3:** Revisar o processo de autoavaliacao (IAA) para reduzir o vies positivo e promover autoconsciencia mais realista.
**Ponto 4:** Adotar abordagem multidimensional na avaliacao de risco, combinando indicadores academicos, comportamentais e psicossociais.
**Ponto 5:** Utilizar a aplicacao Streamlit como ferramenta operacional para a equipe da Passos Magicos monitorar e intervir proativamente.

## Slide 12: Obrigado - Proximos passos
**Titulo:** Obrigado!
**Subtitulo:** Juntos, transformando dados em oportunidades
**Conteudo:** Link do GitHub, link da aplicacao Streamlit, contato da equipe
**Visual:** Fundo com gradiente azul, icones de contato
