"""
Este módulo de configuração centraliza todas as constantes, caminhos e parâmetros
utilizados ao longo do projeto. Ele é projetado para ser importado por outros módulos,
garantindo que todas as partes do código utilizem as mesmas configurações e facilitando
a manutenção futura.
"""
# Atualização: Importação de modelos específicos para a arena de modelos

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# 1. Diretorios e caminho base:

# Captura o diretório raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Pastas de Dados
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "nasa"
NASA_DATA_DIR = RAW_DATA_DIR
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

# 3. Arena de Modelos - Atualização

"""
Dicionário contendo os modelos e hiperparâmetros exatos baseados na Tabela 3 do
artigo de referência. Isso permite iterar facilmente sobre todos os algoritmos durante
o treinamento.
"""
MODELS_TO_TRAIN = {
    "Linear_Regression": LinearRegression(
        fit_intercept=True, 
        n_jobs=None
    ),
    "KNN": KNeighborsRegressor(
        n_neighbors=10
    ),
    "Random_Forest": RandomForestRegressor(
        n_estimators=1200, 
        random_state=42, 
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        objective='reg:squarederror', 
        random_state=42
    ),
    "CatBoost": CatBoostRegressor(
        iterations=1000, 
        learning_rate=0.03, 
        depth=6, 
        loss_function='RMSE', 
        logging_level='Silent', 
        random_state=42
    )
}

# 4. Configuracoes do ML Flow:

"""    
Configurações do MLflow para rastreamento de experimentos, incluindo o nome
do experimento e o URI de rastreamento.
"""

MLFLOW_EXPERIMENT_NAME = "NASA_CMAPSS_RUL_Prediction"
MLFLOW_TRACKING_URI = f"file:///{MLRUNS_DIR.as_posix()}"


# 5. Inicialização das pastas:

def create_directories():
    """ 
    Função para criar as pastas necessárias para armazenar os dados brutos,
    processados e os experimentos do MLflow.
    Utiliza o método `mkdir` do pathlib com `parents=True` para criar toda a hierarquia de pastas,
    e `exist_ok=True` para evitar erros caso as pastas já existam.
    """
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
