import pandas as pd
import mlflow
from shapash.explainer.smart_explainer import SmartExplainer

# Importa as configurações globais
import src.config as config

def generate_explanations(model, X_test, run_id):
    """
    Gera relatórios visuais de XAI usando Shapash, 
    registra no MLflow e extrai as contribuições locais do pior cenário.
    """
    print(f"\n-> Iniciando a auditoria técnica de explicabilidade com Shapash...")
    
    # 1. Instanciando o SmartExplainer e Compilando
    xpl = SmartExplainer(model=model)
    print("-> Compilando contribuições locais e globais...")
    xpl.compile(x=X_test)
    
    # 2. Identificação do Motor em Estado Crítico
    print("-> Extraindo fatores críticos da caixa-preta...")
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    id_critico = y_pred.idxmin()
    rul_predito = y_pred.min()
    
    # Extraímos as variáveis que causaram a pior previsão
    df_contributions = xpl.to_pandas(max_contrib=2)
    fatores = df_contributions.loc[id_critico]
    sensor_1 = fatores['feature_1']
    sensor_2 = fatores['feature_2']
    
    # 3. Exportação de Gráficos HTML Interativos (Plotly Nativo)
    reports_dir = config.BASE_DIR / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("-> Exportando relatórios visuais (HTML)...")
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
        
    print("✅ XAI concluída! Gráficos salvos e registrados.")
    print(f"   Motor mais crítico detectado: Index {id_critico} (RUL Estimado: {rul_predito:.1f} ciclos)")
    print(f"   Sensores responsáveis pela anomalia: {sensor_1} e {sensor_2}")
    
    return id_critico, rul_predito, sensor_1, sensor_2

if __name__ == "__main__":
    print("Este script destina-se a ser orquestrado pelo main.py")