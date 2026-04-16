""" 
Este módulo é responsável por treinar o modelo de previsão de RUL utilizando o dataset da NASA C-Maps. Ele segue as seguintes etapas:
1. Configuração do ambiente MLflow para rastreamento de experimentos.
2. Carregamento do dataset processado.
3. Separação das features e target.
4. Divisão dos dados em conjuntos de treino e teste.
5. Treinamento do modelo Random Forest com os hiperparâmetros definidos no config.py
6. Registro dos hiperparâmetros e do modelo treinado no MLflow para rastreamento e reprodutibilidade.
7. Retorno do modelo treinado e dos dados de teste para a fase de avaliação.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Import das configurações globais
import src.config as config

def run_training_pipeline():

    """
    Função principal para executar o pipeline de treinamento do modelo. Ela carrega os dados processados,
    treina o modelo Random Forest e registra os resultados no MLflow.

    Returns:
    model: O modelo Random Forest treinado.
    X_test: As features do conjunto de teste.
    y_test: Os valores reais do RUL para o conjunto de teste.
    run_id: O ID da execução no MLflow para rastreamento.
    """

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
        print("Iniciando treinamento do modelo Random Forest...")
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
