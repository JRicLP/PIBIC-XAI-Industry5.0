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
import numpy as np
from shapash.explainer.smart_explainer import SmartExplainer

# Importa as configurações globais
import src.config as config

def generate_explanations(model, x_test, y_test, y_pred, test_metadata, run_id=None):
    """
    Gera relatórios visuais de XAI usando Shapash,
    registra no MLflow e extrai as contribuições locais do pior cenário.

    Args:
    model: O modelo treinado para o qual as explicações serão geradas.
    x_test: O conjunto de teste usado para compilar as explicações.
    run_id: O ID da execução no MLflow para registrar os artefatos.
    Returns:
    motor_critico: Identificação operacional do motor crítico.
    sensor_1: O sensor mais responsável pela degradação do motor crítico.
    sensor_2: O segundo sensor mais responsável pela degradação do motor crítico.
    """
    print("Iniciando a auditoria técnica de explicabilidade com Shapash...")

    # 1. Instanciando o SmartExplainer e Compilando
    xpl = SmartExplainer(model=model, features_dict=config.FEATURES_DICT_NASA)
    y_pred = np.asarray(y_pred)
    xai_x_test = x_test
    xai_test_metadata = test_metadata

    if not hasattr(model, 'estimators_') and len(x_test) > 200:
        print("Modelo sem estimators_. Usando amostra de 200 instancias para evitar lentidao no Shapash.")
        xai_x_test = x_test.sample(n=200, random_state=config.SEED)
        sample_positions = x_test.index.get_indexer(xai_x_test.index)
        y_pred = y_pred[sample_positions]
        xai_test_metadata = test_metadata.iloc[sample_positions].reset_index(drop=True)

    print("Compilando contribuicoes locais e globais...")
    try:
        xpl.compile(x=xai_x_test)
    except Exception as exc:
        print(f"Aviso: Shapash falhou ao compilar explicabilidade ({exc}).")
        return None, None, None, None

    # 2. Identificação do Motor em Estado Crítico
    print("Extraindo fatores críticos da caixa-preta...")
    # Atualização: Identificamos o motor com a pior previsão de RUL (menor valor previsto)
    if hasattr(model, 'estimators_'):
        # Calcula o Desvio Padrão entre as árvores para identificar a incerteza
        preds_por_arvore = np.array([
            arvore.predict(xai_x_test) for arvore in model.estimators_
        ])
        incerteza = preds_por_arvore.std(axis=0)
        # Combina a previsão média com a incerteza para identificar o motor mais crítico
        risco_combinado = y_pred - incerteza  # Penaliza previsões com alta incerteza
        posicao_critica = int(np.argmin(risco_combinado))
        id_critico = posicao_critica
        print("Motor crítico identificado com base na combinação de RUL previsto e incerteza.")
    else:
        id_critico = int(np.argmin(y_pred))
        incerteza = None
        print("Motor crítico identificado com base na previsão de RUL.")

    indice_local = xai_x_test.index[id_critico]
    engine_id_real = xai_test_metadata.iloc[id_critico]["engine_id"]
    time_cycle = xai_test_metadata.iloc[id_critico]["time_cycle"]
    rul_predito = y_pred[id_critico]
    motor_critico = {
        "engine_id": engine_id_real,
        "time_cycle": time_cycle,
        "rul_predito": rul_predito
    }
    # Extraímos as variáveis que causaram a pior previsão
    df_contributions = xpl.to_pandas(max_contrib=2)
    if indice_local in df_contributions.index:
        fatores = df_contributions.loc[indice_local]
    else:
        fatores = df_contributions.iloc[id_critico]
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
    fig_local = xpl.plot.local_plot(index=indice_local)
    local_path = reports_dir / "shapash_local_critico.html"
    fig_local.write_html(str(local_path))

    # 4. Registro dos Gráficos no MLflow
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(global_path), artifact_path="xai_reports")
            mlflow.log_artifact(str(local_path), artifact_path="xai_reports")

    print("  XAI concluída! Gráficos salvos e registrados.")
    print(
        f" Motor mais crítico detectado: engine_id {engine_id_real}, "
        f"ciclo {time_cycle} (RUL Estimado: {rul_predito:.1f} ciclos)"
    )
    print(f" Sensores responsáveis pela anomalia: {sensor_1} e {sensor_2}")

    return motor_critico, sensor_1, sensor_2, df_contributions

if __name__ == "__main__":
    print("Este script destina-se a ser orquestrado pelo main.py")
