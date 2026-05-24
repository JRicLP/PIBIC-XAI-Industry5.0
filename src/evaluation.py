""" 
Este módulo é responsável por avaliar o desempenho dos modelos treinados usando métricas de regressão
(MAE, RMSE, R²) e gerar gráficos de validação individualizados (Real vs Previsto e Distribuição dos Resíduos).
As métricas e gráficos são registados no MLflow para cada modelo avaliado. O módulo é projetado para ser
chamado a partir do main.py, onde os modelos treinados e os dados de teste são passados para avaliação.

Passos principais:
1. Geração das Previsões: O modelo treinado é usado para gerar previsões com base no conjunto de teste.
2. Cálculo das Métricas Matemáticas: As métricas de regressão (MAE, RMSE, R²) são calculadas para
quantificar o desempenho do modelo.
3. Registo no MLflow: As métricas calculadas são registadas na mesma execução do MLflow onde o modelo foi treinado.
4. Geração de Gráficos de Validação: São criados gráficos de dispersão (Real vs Previsto) e histogramas
(Distribuição dos Resíduos) para análise visual do desempenho do modelo.
5. Registo dos Artefactos no MLflow: Os gráficos gerados são salvos localmente e registados como
artefactos no MLflow para referência futura.
"""

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config

# Atualização: Adicionando o parâmetro NASA scoring function para avaliação adicional

def nasa_score(y_true, y_pred):
    """ 
    Calcula a função de pontuação personalizada da NASA para avaliar as
    previsões de RUL, penalizando superestimações e subestimações de forma diferente.
    Args:
    y_true: Os valores reais do RUL.
    y_pred: Os valores previstos do RUL.
    Returns:
    score: A pontuação média calculada usando a função de pontuação da NASA.
    """
    diff = y_pred - y_true # Positivo = Superestimação, Negativo = Subestimação
    score = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    return float(np.sum(score))

def calculate_mae_by_rul_range(y_true, y_pred):
    """
    Calcula o MAE para diferentes faixas de RUL (0-25, 25-75, 75+), permitindo uma análise mais granular
    do desempenho do modelo em diferentes estágios de vida útil dos equipamentos.
    Args:
    y_true: Os valores reais do RUL.
    y_pred: Os valores previstos do RUL.
    Returns:
    mae_by_range: Um dicionário contendo o MAE para cada faixa de RUL.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ranges = {
        "mae_critico_rul_0_25": y_true <= 25,
        "mae_medio_rul_25_75": (y_true > 25) & (y_true <= 75),
        "mae_distante_rul_75_plus": y_true > 75,
    }

    return {
        name: (
            float(mean_absolute_error(y_true[mask], y_pred[mask]))
            if np.any(mask)
            else np.nan
        )
        for name, mask in ranges.items()
    }

def evaluate_model(model, x_test, y_test, run_id, model_name, y_pred=None):
    """
    Calcula as métricas de regressão, gera gráficos de validação individualizados
    e regista os resultados e artefactos no MLflow.

    Args:
    model: O modelo treinado a ser avaliado.
    x_test: O conjunto de teste usado para gerar previsões.
    y_test: Os valores reais do RUL para o conjunto de teste.
    run_id: O ID da execução no MLflow para registar as métricas e artefatos.
    model_name: O nome do modelo (string) para nomear os gráficos exportados.
    
    Returns:    
    mae: O erro absoluto médio entre as previsões e os valores reais.
    rmse: A raiz do erro quadrático médio entre as previsões e os valores reais
    r2: O coeficiente de determinação (R²) que indica a qualidade do ajuste do modelo.
    """
    print(f"Início da avaliação estatística (Run ID: {run_id})...")
    
    # 1. Geração das Previsões
    if y_pred is None:
        y_pred = model.predict(x_test)
    y_pred = np.asarray(y_pred)
    
    # 2. Cálculo das Métricas Matemáticas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    nasa_score_value = nasa_score(y_test, y_pred) # Nova métrica de avaliação personalizada da NASA
    mae_by_range = calculate_mae_by_rul_range(y_test, y_pred)

    print(f"   NASA Score (Assimétrico, Menor = Melhor): {nasa_score_value:.2f}")
    print(f"   MAE  (Erro Absoluto Médio): {mae:.2f} ciclos")
    print(f"   RMSE (Raiz do Erro Quadrático Médio): {rmse:.2f} ciclos")
    print(f"   R²   (Coeficiente de Determinação): {r2:.4f}")
    print(f"   MAE critico (RUL 0-25): {mae_by_range['mae_critico_rul_0_25']:.2f} ciclos")
    print(f"   MAE medio   (RUL 25-75): {mae_by_range['mae_medio_rul_25_75']:.2f} ciclos")
    print(f"   MAE distante (RUL 75+): {mae_by_range['mae_distante_rul_75_plus']:.2f} ciclos")
    
    # 3. Registo no MLflow (reabrindo a Run do treino)
    with mlflow.start_run(run_id=run_id):
        # Registo de métricas
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("nasa_score", nasa_score_value)
        for metric_name, metric_value in mae_by_range.items():
            if not np.isnan(metric_value):
                mlflow.log_metric(metric_name, metric_value)
        # 4. Geração de Gráficos de Validação
        # Garantindo que a pasta de exportação local existe
        plots_dir = config.BASE_DIR / "docs" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Gráfico 1: Real vs Previsto
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='#1f77b4') # Azul padrão
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("RUL Real (Ciclos de Vida)")
        plt.ylabel("RUL Previsto (Ciclos de Vida)")
        plt.title(f"Validação do Modelo ({model_name}): RUL Real vs Previsto")
        plt.grid(True, linestyle='--', alpha=0.7)
        scatter_path = plots_dir / f"{model_name}_real_vs_previsto.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico 2: Distribuição dos Resíduos (Erros)
        residuos = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuos, bins=50, kde=True, color='#9467bd') # Roxo
        plt.axvline(0, color='r', linestyle='--', lw=2)
        plt.xlabel("Erro Residual (Ciclos)")
        plt.ylabel("Frequência")
        plt.title(f"Análise de Erro ({model_name}): Distribuição dos Resíduos")
        plt.grid(True, linestyle='--', alpha=0.7)
        residuos_path = plots_dir / f"{model_name}_distribuicao_residuos.png"
        plt.savefig(residuos_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Registo dos Artefactos (Gráficos) no MLflow
        mlflow.log_artifact(str(scatter_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(residuos_path), artifact_path="evaluation_plots")
        
    print(f"Avaliação estatística para {model_name} concluída. Registado no MLflow.")
    
    return mae, rmse, r2, nasa_score_value

if __name__ == "__main__":
    print("Este guia destina-se a ser orquestrado pelo main.py e não deve ser executado diretamente.")
