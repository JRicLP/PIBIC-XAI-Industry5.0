import os
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

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00273/CMAPSSData.zip"
TRAIN_FILE = "train_FD001.txt"
TEST_FILE = "test_FD001.txt"
RUL_FILE = "RUL_FD001.txt"
PROCESSED_FILE_NAME = "nasa_cmapss_fd001_processed.csv"

# Nomenclatura oficial das colunas:
INDEX_COLUMNS = ['engine_id', 'time_cycle']
SETTING_COLUMNS = ['setting_1', 'setting_2', 'setting_3']
SENSOR_COLUMNS = [f'sensor_{i}' for i in range(1, 22)]

# Junção de todas as colunas para a leitura do .txt original
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# 3. Hiperparametros do modelo (Random Forest):

# Dicionário centralizando os parâmetros para facilitar o rastreamento no MLflow
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,       # Evita overfitting nas árvores
    'random_state': 42,    # Garante a reprodutibilidade estatística
    'n_jobs': -1           # Utiliza todos os núcleos do processador
}

# 4. Configuracoes do ML Flow:

MLFLOW_EXPERIMENT_NAME = "NASA_CMAPSS_RUL_Prediction"
MLFLOW_TRACKING_URI = f"file:///{MLRUNS_DIR.as_posix()}"


# 5. Inicialização das pastas:

"""Cria a estrutura de pastas caso não exista."""

def create_directories():
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Executa a criação de pastas ao importar o config
create_directories()