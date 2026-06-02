# Resultados - NASA C-MAPSS FD001

## 1. Descricao do problema

O NASA C-MAPSS FD001 simula a degradacao progressiva de motores turbofan em uma unica condicao operacional e um unico modo de falha. A tarefa do projeto e estimar a vida util restante, ou RUL, a partir de ciclos operacionais e sensores do motor.

Em manutencao preditiva, o erro de RUL tem impacto operacional direto: superestimar a vida restante pode atrasar uma intervencao critica, enquanto subestimar a vida restante pode antecipar paradas e elevar custo. Por isso, alem de MAE, RMSE e R2, o pipeline tambem registra o NASA Score, que penaliza de modo assimetrico erros de previsao.

## 2. Metodologia

O pipeline usa o protocolo oficial do FD001: treinamento em `train_FD001.txt` e avaliacao no ultimo ciclo disponivel de cada motor de `test_FD001.txt`, comparado contra `RUL_FD001.txt`.

O alvo RUL e limitado em 125 ciclos, seguindo a abordagem piecewise linear comum na literatura de C-MAPSS. Sensores de baixa variancia sao removidos antes do treinamento: `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18` e `sensor_19`. Os sensores restantes sao normalizados com `MinMaxScaler`.

O torneio atual separa baselines fixos e modelos avancados. Linear Regression e KNN entram como baselines interpretaveis. Random Forest, XGBoost e CatBoost passam por tuning com Optuna antes do torneio final, usando `GroupKFold(n_splits=5)` agrupado por `engine_id` e minimizando o MAE medio de validacao cruzada. O teste oficial NASA continua reservado para a avaliacao final, depois do tuning.

## 3. Resultados quantitativos

Resultado mais recente registrado em `mlruns`, tambem estruturado em `docs/nasa/resultados.json`. Este resultado foi gerado antes da execucao completa do tuning Optuna:

| Modelo campeao | MAE | RMSE | R2 | NASA Score | MLflow run |
| --- | ---: | ---: | ---: | ---: | --- |
| CatBoost | 11.7874 | 16.7850 | 0.8246 | 772.8196 | `2f85d8faca744cf2af35cfe5eec64c00` |

MAE por faixa de RUL:

| Faixa de RUL | MAE |
| --- | ---: |
| 0-25 ciclos | 9.0052 |
| 25-75 ciclos | 13.2688 |
| 75+ ciclos | 12.0910 |

A faixa critica, entre 0 e 25 ciclos restantes, tem o menor MAE do resultado registrado. Isso e positivo para manutencao, pois a regiao de falha iminente e justamente a mais sensivel para tomada de decisao.

## 4. Analise das explicacoes XAI

A explicabilidade e gerada por `src/explainability.py` com Shapash. O relatorio global mostra quais sensores mais influenciam a regressao de RUL, enquanto o relatorio local identifica os dois principais fatores do motor mais critico detectado no teste oficial.

Os sensores do C-MAPSS representam grandezas fisicas do turbofan, como temperaturas em estagios de compressor e turbina, pressoes, rotacoes corrigidas e sangrias de resfriamento. Quando sensores ligados a temperatura, pressao ou velocidade passam a dominar a explicacao local, a leitura operacional e que o modelo esta associando a queda de RUL a mudancas fisicas plausiveis no ciclo termodinamico do motor.

Artefatos gerados pelo pipeline:

- `docs/reports/shapash_global.html`
- `docs/reports/shapash_local_critico.html`
- `docs/plots/<modelo>_real_vs_previsto.png`
- `docs/plots/<modelo>_distribuicao_residuos.png`

## 5. Exemplo de laudo gerado

Quando `GEMINI_API_KEY` nao esta configurada, o pipeline gera um laudo simulado para manter a reproducibilidade local:

```text
Atencao mecanico: O motor identificado apresenta falha iminente em aproximadamente N ciclos.
Inspecione imediatamente os componentes relacionados ao sensor principal e verifique tambem a integridade do sensor secundario.
```

Com a chave configurada, `src/llm_assistant.py` envia ao Gemini o motor critico, o ciclo operacional, o RUL predito e os dois sensores mais relevantes. A camada LLM nao altera a predicao; ela traduz a saida matematica do XAI em um texto curto, orientado a acao, para apoiar a inspeccao do operador.

## 6. Limitacoes e trabalhos futuros

O tuning com Optuna foi implementado com budget padrao de 30 trials por modelo avancado. Os studies sao salvos em `docs/optuna_studies/` e podem ser retomados em caso de interrupcao.

Os valores consolidados do torneio tunado devem substituir a tabela acima apos a proxima execucao completa de `python -m src.main`, pois o resultado atual ainda reflete o ultimo run registrado antes desta etapa.

Como trabalhos futuros, recomenda-se avaliar modelos sequenciais, comparar diferentes caps de RUL, executar estudo de sensibilidade por motor e validar a utilidade dos laudos com usuarios tecnicos.

## 7. Referencias

- Saxena, A. and Goebel, K. (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames Prognostics Data Repository.
- Heimes, F. O. (2008). Recurrent neural networks for remaining useful life estimation.
- Babu, G. S., Zhao, P., and Li, X. L. (2016). Deep convolutional neural network based regression approach for estimation of remaining useful life.
- Shapash documentation for model explainability dashboards.
- MLflow documentation for experiment tracking.
