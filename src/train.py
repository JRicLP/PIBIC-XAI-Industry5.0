"""
Este módulo é responsável por treinar os modelos definidos na arena de modelos do config.py,
utilizando o pipeline de treinamento que inclui o registro no MLflow. Ele é projetado para
ser flexível, permitindo a adição de novos modelos e hiperparâmetros sem a necessidade de alterar
a estrutura principal do código. O módulo também inclui um teste isolado para garantir que o process
de treinamento funcione corretamente.

O fluxo de treinamento é o seguinte:
1. Carregamento dos dados processados.
2. Separação dos dados em conjuntos de treinamento e teste.
3. Treinamento do modelo utilizando a instância e os hiperparâmetros definidos no config.py.
4. Registro do modelo treinado no MLflow, incluindo o nome do modelo e os hiperparâmetros utilizados.
5. Retorno do modelo treinado, conjunto de teste e run_id para avaliação posterior.
"""

# Imports
import mlflow
import mlflow.sklearn
import pandas as pd
from src import config

# Atualização - Flexibilidade para treinar diferentes modelos da arena de modelos definida no config.py
def run_training_pipeline(model_name, model_instance):
    """
    Executa o pipeline de treinamento do modelo, incluindo o registro no MLflow de
    acordo com o modelo e hiperparâmetros definidos no config.py.

    Args:
        model_name (str): Nome do modelo para registro no MLflow.
        model_instance: Instância do modelo a ser treinado, configurada com os hiperparâmetros do config.py.
    Returns:
        model_instance: O modelo treinado.
    """
    print(f"\nConectando ao MLflow em: {config.MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # 1 e 2. Carregamento e Separação (Mantém o seu código original intacto)
    data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_FILE_NAME
    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}. Execute o pipeline de dados primeiro.")

    official_test_path = config.PROCESSED_DATA_DIR / config.PROCESSED_OFFICIAL_TEST_FILE_NAME
    if not official_test_path.exists():
        raise FileNotFoundError(
            f"Arquivo nÃ£o encontrado: {official_test_path}. Execute o pipeline de dados primeiro."
        )

    df_train = pd.read_csv(data_path)
    df_test = pd.read_csv(official_test_path)

    # Protocolo oficial NASA: treino em train_FD001 e teste no ultimo ciclo de test_FD001.
    x_train = df_train.drop(columns=['engine_id', 'time_cycle', 'RUL'])
    y_train = df_train['RUL']
    x_test = df_test.drop(columns=['engine_id', 'time_cycle', 'RUL'])
    y_test = df_test['RUL']
    test_metadata = df_test[
        ["engine_id", "time_cycle"]
    ].reset_index(drop=True)


    # 3. Inicialização do Registro no MLflow
    with mlflow.start_run(run_name=model_name) as run:
        print(f"Iniciando treinamento do modelo: {model_name}...")

        # Treinamento dinâmico
        model_instance.fit(x_train, y_train)

        # 4. Registro no MLflow
        mlflow.sklearn.log_model(model_instance, model_name)
        print(f"Treinamento concluído. Run ID: {run.info.run_id}")

        y_pred = model_instance.predict(x_test)
        return model_instance, x_test, y_test, y_pred, test_metadata, run.info.run_id

if __name__ == "__main__":
    # Teste isolado para garantir que o pipeline de treinamento funcione corretamente

    # Escolhe o primeiro modelo da arena de modelos para teste
    nome_padrao, instancia_padrao = list(config.MODELS_TO_TRAIN.items())[0]

    try:
        run_training_pipeline(nome_padrao, instancia_padrao)
    except Exception as e:
        print(f"\n[Erro no Teste Isolado] Não foi possível executar o treino: {e}")
