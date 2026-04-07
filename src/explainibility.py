import pandas as pd
import mlflow
from shapash.explainer.smart_explainer import SmartExplainer

# Import das configurações globais:
import src.config as config

def generate_explanations(model, X_test, run_id):

    """
    Descrição geral:

    - Geração do relatório de explicabilidade (XAI) usando Shapash. 
    - Registra no MLflow e extrai as contribuições locais do pior cenário.
    """

    print("Iniciando a auditoria técnica de explicabilidade com Shapash...")
    
    # 1. Instanciando o SmartExplainer
    xpl = SmartExplainer()
    
    # 2. Compilação
    print("Compilando contribuições locais e globais...")
    xpl.compile(x=X_test, model=model)
    
    # 3. Geração e Salvamento do Dashboard HTML estático
    reports_dir = config.BASE_DIR / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "shapash_dashboard.html"
    
    print("Exportando relatório HTML...")
    # Em um pipeline automatizado, salvamos o HTML ao invés de subir o app web (run_app)
    xpl.save_html(output_file=str(html_path))
    
    # 4. Registro do Dashboard no MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(html_path), artifact_path="xai_reports")
        
    # 5. Preparação para a Indústria 5.0 (Extração de Contexto para a LLM):

    # Identifica-se a amostra (motor) com o RUL mais crítico no teste
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    id_critico = y_pred.idxmin()
    rul_predito = y_pred.min()
    
    # Extraí-se do Shapash os dois sensores que mais impactaram essa previsão crítica
    print("Extraindo fatores críticos do modelo black-box...")
    df_contributions = xpl.to_pandas(max_contrib=2)
    fatores = df_contributions.loc[id_critico]
    sensor_1 = fatores['feature_1']
    sensor_2 = fatores['feature_2']
    
    print("XAI concluída. Dashboard salvo e registrado.")
    print(f"Motor mais crítico detectado: Index {id_critico} (RUL Estimado: {rul_predito:.1f} ciclos)")
    print(f"Sensores responsáveis pela anomalia: {sensor_1} e {sensor_2}")
    
    # Retorna os dados traduzidos pela IA para usarmos no prompt do Gemini
    return id_critico, rul_predito, sensor_1, sensor_2

if __name__ == "__main__":
    print("Este script destina-se a ser orquestrado pelo main.py")