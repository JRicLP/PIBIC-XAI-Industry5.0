""" 
Este módulo é responsável por gerar relatórios visuais de XAI utilizando
a biblioteca Shapash. Ele segue as seguintes etapas:

1. Instanciação do SmartExplainer com o modelo treinado.
2. Compilação das contribuições locais e globais para o conjunto de teste.
3. Identificação do motor em estado crítico (pior previsão de RUL) 
e extração dos fatores responsáveis.
4. Exportação de gráficos interativos em HTML para análise visual.
5. Registro dos gráficos como artefatos no MLflow para rastreamento e reprodutibilidade.
6. Retorno dos insights críticos para a fase de Assistente LLM, permitindo a tradução
das explicações para recomendações acionáveis para os operadores de pista.
"""

import mlflow
import pandas as pd
import numpy as np
from shapash.explainer.smart_explainer import SmartExplainer

# Importa as configurações globais
import src.config as config

def generate_explanations(model, x_test, run_id):
    """
    Gera relatórios visuais de XAI usando Shapash, 
    registra no MLflow e extrai as contribuições locais do pior cenário.

    Args:
    model: O modelo treinado para o qual as explicações serão geradas.
    x_test: O conjunto de teste usado para compilar as explicações.
    run_id: O ID da execução no MLflow para registrar os artefatos.
    Returns:
    id_critico: O índice do motor com a pior previsão de RUL.
    rul_predito: O valor previsto de RUL para o motor crítico.
    sensor_1: O sensor mais responsável pela degradação do motor crítico.
    sensor_2: O segundo sensor mais responsável pela degradação do motor crítico.
    """
    print("Iniciando a auditoria técnica de explicabilidade com Shapash...")
    
    # 1. Instanciando o SmartExplainer e Compilando
    xpl = SmartExplainer(model=model)
    print("Compilando contribuições locais e globais...")
    xpl.compile(x=x_test)
    
    # 2. Identificação do Motor em Estado Crítico
    print("Extraindo fatores críticos da caixa-preta...")
    y_pred = pd.Series(model.predict(x_test), index=x_test.index)

    # Atualização: Identificamos o motor com a pior previsão de RUL (menor valor previsto)
    if hasattr(model, 'estimators_'):
        # Calcula o Desvio Padrão entre as árvores para identificar a incerteza
        preds_por_arvore = np.array([
            arvore.predict(x_test) for arvore in model.estimators_  
        ])
        incerteza = pd.Series(preds_por_arvore.std(axis=0), index=x_test.index)
        # Combina a previsão média com a incerteza para identificar o motor mais crítico
        risco_combinado = y_pred - incerteza  # Penaliza previsões com alta incerteza
        id_critico = risco_combinado.idxmin()
        print(f"Motor crítico identificado com base na combinação de RUL previsto e incerteza: Index {id_critico}")
    else:
        id_critico = y_pred.idxmin()
        incerteza = None
        print(f"Motor crítico identificado com base na previsão de RUL: Index {id_critico}")

    rul_predito = y_pred[id_critico]
    # Extraímos as variáveis que causaram a pior previsão
    df_contributions = xpl.to_pandas(max_contrib=2)
    fatores = df_contributions.loc[id_critico]
    sensor_1 = fatores['feature_1']
    sensor_2 = fatores['feature_2']

    # 3. Exportação de Gráficos HTML Interativos (Plotly Nativo)
    reports_dir = config.BASE_DIR / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    print("Exportando relatórios visuais (HTML)...")

    # Gráfico Global de Importância das Features
    fig_global = xpl.plot.features_importance()
    global_path = reports_dir / "shapash_global.html"
    fig_global.write_html(str(global_path))    
    # Gráfico Local específico do motor prestes a falhar
    fig_local = xpl.plot.local_plot(index=id_critico)
    local_path = reports_dir / "shapash_local_critico.html"
    fig_local.write_html(str(local_path))
    
    # 4. Registro dos Gráficos no MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(global_path), artifact_path="xai_reports")
        mlflow.log_artifact(str(local_path), artifact_path="xai_reports")
        
    print("  XAI concluída! Gráficos salvos e registrados.")
    print(f" Motor mais crítico detectado: Index {id_critico} (RUL Estimado: {rul_predito:.1f} ciclos)")
    print(f" Sensores responsáveis pela anomalia: {sensor_1} e {sensor_2}")
    
    return id_critico, rul_predito, sensor_1, sensor_2, df_contributions

if __name__ == "__main__":
    print("Este script destina-se a ser orquestrado pelo main.py")
