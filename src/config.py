"""
Este módulo centraliza todas as configurações globais do projeto, incluindo
caminhos de diretórios,parâmetros do modelo e configurações do MLflow.

Ao manter essas informações em um único arquivo, garantimos que o código seja mais organizado,
fácil de manter e que as mudanças futuras possam ser feitas de forma rápida e sem a necessidade
de modificar múltiplos arquivos.
"""

from pathlib import Path

# 1. Diretorios e caminho base:

# Captura o diretório raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Pastas de Dados
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "nasa"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "nasa"

# Pasta do MLflow
MLRUNS_DIR = BASE_DIR / "mlruns"

# 2. Configuracoes do Dataset (Nasa C-Maps)

"""
Configurações específicas para o dataset da NASA C-Maps, incluindo URLs,
nomes de arquivos e estrutura das colunas.

Essas configurações são utilizadas em todo o pipeline para garantir consistência
e facilitar a manutenção do código. 

A centralização dessas informações em um único arquivo de configuração permite 
que ajustes futuros sejam feitos de forma rápida e sem a necessidade de modificar múltiplos arquivos.
"""

DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
TRAIN_FILE = "train_FD001.txt"
TEST_FILE = "test_FD001.txt"
RUL_FILE = "RUL_FD001.txt"
PROCESSED_FILE_NAME = "nasa_cmapss_fd001_processed.csv"

# Nomenclatura oficial das colunas:
INDEX_COLUMNS = ["engine_id", "time_cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

# Junção de todas as colunas para a leitura do .txt original
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# 3. Hiperparametros do modelo (Random Forest):

"""
Hiperparâmetros do modelo Random Forest centralizados em um dicionário para facilitar
o rastreamento no MLflow e a manutenção do código.

Ao manter esses parâmetros em um único local, garantimos que as mudanças futuras possam ser
feitas de forma rápida e sem a necessidade de modificar múltiplos arquivos, além de facilitar
a comparação entre diferentes experimentos.
"""

# Dicionário centralizando os parâmetros para facilitar o rastreamento no MLflow
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,  # Evita overfitting nas árvores
    "random_state": 42,  # Garante a reprodutibilidade estatística
    "n_jobs": -1,  # Utiliza todos os núcleos do processador
}

# 4. Configuracoes do ML Flow:

"""    
Configurações do MLflow para rastreamento de experimentos, incluindo o nome do experimento e o URI de rastreamento.
"""

MLFLOW_EXPERIMENT_NAME = "NASA_CMAPSS_RUL_Prediction"
MLFLOW_TRACKING_URI = f"file:///{MLRUNS_DIR.as_posix()}"


# 5. Inicialização das pastas:

def create_directories():
    """ 
    Função para criar as pastas necessárias para armazenar os dados brutos, processados e os experimentos do MLflow.
    Utiliza o método `mkdir` do pathlib com `parents=True` para criar toda a hierarquia de pastas, e `exist_ok=True` para evitar erros caso as pastas já existam.
    """
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Executa a criação de pastas ao importar o config
create_directories()
