import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow

# Importa as configurações globais
import src.config as config

def evaluate_model(model, X_test, y_test, run_id):
    """
    Calcula as métricas de regressão, gera gráficos de validação
    e regista os resultados e artefactos no MLflow.
    """
    print(f"Início da avaliação estatística (Run ID: {run_id})...")
    
    # 1. Geração das Previsões
    y_pred = model.predict(X_test)
    
    # 2. Cálculo das Métricas Matemáticas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"   MAE  (Erro Absoluto Médio): {mae:.2f} ciclos")
    print(f"   RMSE (Raiz do Erro Quadrático Médio): {rmse:.2f} ciclos")
    print(f"   R²   (Coeficiente de Determinação): {r2:.4f}")
    
    # 3. Registo no MLflow (reabrindo a Run do treino)
    with mlflow.start_run(run_id=run_id):
        # Registo de métricas
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # 4. Geração de Gráficos de Validação
        # Garantir que a pasta de exportação local existe (ideal para o relatório do PIBIC)
        plots_dir = config.BASE_DIR / "docs" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        #Gráfico 1: Real vs Previsto
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='#1f77b4') # Azul padrão
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("RUL Real (Ciclos de Vida)")
        plt.ylabel("RUL Previsto (Ciclos de Vida)")
        plt.title("Validação do Modelo: RUL Real vs Previsto")
        plt.grid(True, linestyle='--', alpha=0.7)
        scatter_path = plots_dir / "real_vs_previsto.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        #Gráfico 2: Distribuição dos Resíduos (Erros)
        residuos = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuos, bins=50, kde=True, color='#9467bd') # Roxo
        plt.axvline(0, color='r', linestyle='--', lw=2)
        plt.xlabel("Erro Residual (Ciclos)")
        plt.ylabel("Frequência")
        plt.title("Análise de Erro: Distribuição dos Resíduos")
        plt.grid(True, linestyle='--', alpha=0.7)
        residuos_path = plots_dir / "distribuicao_residuos.png"
        plt.savefig(residuos_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Registo dos Artefactos (Gráficos) no MLflow
        mlflow.log_artifact(str(scatter_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(residuos_path), artifact_path="evaluation_plots")
        
    print("Avaliação estatística concluída. Métricas e gráficos registados no MLflow.")
    
    return mae, rmse, r2

if __name__ == "__main__":
    print("Este guião destina-se a ser orquestrado pelo main.py")