# Imports

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Import das configurações globais
import src.config as config

"""
Carrega os dados processados, treina o modelo Random Forest 
e registra o experimento no MLflow.
"""

def run_training_pipeline():

    print(f"Conectando ao MLflow em: {config.MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # 1. Carregamento dos Dados Processados
    data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_FILE_NAME
    
    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}. Execute o pipeline de dados primeiro.")
    
    print("Carregando dataset processado...")
    df = pd.read_csv(data_path)

    # 2. Separação de Features (X) e Target (y)
    # Exclui colunas de identificação e mantém apenas sensores e settings para prever o RUL
    X = df.drop(columns=['engine_id', 'time_cycle', 'RUL'])
    y = df['RUL']

    # Divisão em treino e teste (mantendo o rigor estatístico)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.MODEL_PARAMS['random_state'])

    # 3. Inicialização do Registro no MLflow
    with mlflow.start_run() as run:
        print("-> Iniciando treinamento do modelo Random Forest...")
        
        # Instancia o modelo com os hiperparâmetros do config.py
        model = RandomForestRegressor(
            n_estimators=config.MODEL_PARAMS['n_estimators'],
            max_depth=config.MODEL_PARAMS['max_depth'],
            random_state=config.MODEL_PARAMS['random_state'],
            n_jobs=config.MODEL_PARAMS['n_jobs']
        )
        
        # Treinamento
        model.fit(X_train, y_train)
        
        # 4. Registro no MLflow
        # Registra os hiperparâmetros
        mlflow.log_params(config.MODEL_PARAMS)
        
        # Registra o modelo treinado
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Treinamento concluído. Run ID: {run.info.run_id}")
        
        # Retorna o modelo e os dados de teste para a próxima fase (Avaliação)
        return model, X_test, y_test, run.info.run_id

if __name__ == "__main__":
    # Permite testar o script isoladamente
    run_training_pipeline()