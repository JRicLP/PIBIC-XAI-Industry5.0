"""
Este módulo é responsável por todo o processo de Extração, Transformação e
Carga (ETL) dos dados brutos do dataset CMAPSS da NASA. Ele inclui as seguintes etapas:

1. Download e Extração: Baixa o dataset da NASA e extrai os arquivos necessários.
2. Leitura dos Dados: Carrega os arquivos de treino em um DataFrame do Pandas, 
aplicando as nomenclaturas das colunas definidas no config.py.
3. Cálculo do RUL: Calcula a Vida Útil Restante (RUL) para cada motor com
base no ciclo máximo registrado para cada motor.
4. Salvamento do Dataset Processado: Salva o DataFrame final com as features e
a coluna de RUL em um arquivo CSV para uso posterior no treinamento do modelo.
5. Orquestração do Pipeline: A função run_etl_pipeline() coordena todas as
etapas do processo, garantindo que os dados estejam prontos para a fase de treinamento.
6. Teste Isolado: Permite a execução do pipeline de dados de forma isolada para
validação e desenvolvimento independente.
"""

# Imports
import urllib.request
import zipfile
import pandas as pd
from src import config
from sklearn.preprocessing import MinMaxScaler

def download_and_extract_data():
    """
    Baixa e extrai o dataset CMAPSS da NASA caso os arquivos brutos não existam no diretório local.
    """

    zip_path = config.RAW_DATA_DIR / "CMAPSSData.zip"
    # Verificando se o arquivo de treino já foi extraído
    required_files = [
        config.RAW_DATA_DIR / config.TRAIN_FILE,
        config.RAW_DATA_DIR / config.TEST_FILE,
        config.RAW_DATA_DIR / config.RUL_FILE,
    ]
    if all(path.exists() for path in required_files):
        print("Dataset bruto já existente. Pulando download.")
        return

    print("Baixando dataset da NASA (CMAPSS)...")
    urllib.request.urlretrieve(config.DATASET_URL, zip_path)
    print("Extraindo arquivos...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(config.RAW_DATA_DIR)
    print("Download e extração concluídos.")

def load_data(file_name=config.TRAIN_FILE):
    """
    Carrega o arquivo de treino txt delimitado por espaços e aplica 
    as nomenclaturas das colunas definidas no config.py.
    """

    train_path = config.RAW_DATA_DIR / file_name
    print("Carregando dados brutos...")
    
    # O dataset original usa espaços como delimitadores e não possui cabeçalho
    df_train = pd.read_csv(train_path, sep=r'\s+', header=None, names=config.ALL_COLUMNS)
    return df_train

def calculate_rul(df):
    """
    Calcula a Vida Útil Restante (RUL) para cada motor com base no ciclo máximo.
    RUL = (Ciclo Máximo do Motor) - (Ciclo Atual)

    Args:
    df: DataFrame contendo os dados brutos com as colunas de identificação, settings e sensores.
    Returns:
    df: DataFrame atualizado com uma nova coluna 'RUL' representando a vida útil restante de cada motor.
    """

    print("Calculando o RUL (Remaining Useful Life)...")
    
    # Encontra o ciclo de vida máximo para cada motor
    max_cycles = df.groupby('engine_id')['time_cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    
    # Faz o merge com o dataset original
    df = df.merge(max_cycles, on=['engine_id'], how='left')
    
    # Calcula o RUL contínuo
    df['RUL'] = df['max_cycle'] - df['time_cycle']
    df['RUL'] = df['RUL'].clip(upper=config.RUL_CAP)
    
    # Remove a coluna auxiliar 'max_cycle' para manter o dataset limpo
    df.drop('max_cycle', axis=1, inplace=True)
    
    return df

# Atualização: Remoção de sensores de baixa variância e normalização dos dados

def process_data(df, scaler=None):
    """ 
    Processa os dados removendo sensores de baixa variância e aplicando normalização.
    Args:
        df: DataFrame contendo os dados brutos com as colunas de identificação, settings e sensores.
    Returns:
        df: DataFrame processado com sensores de baixa variância removidos e dados normalizados.
    """
    print("Pré-processamento: Removendo sensores de baixa variância e aplicando normalização")
    SENSORES_BAIXA_VARIANCIA = [
        'sensor_1', 'sensor_5', 'sensor_10',
        'sensor_16', 'sensor_18', 'sensor_19'
    ]
    df = df.drop(columns=SENSORES_BAIXA_VARIANCIA)

    print("Normalização dos Sensores com MinMaxScaler")
    sensors_cols = [col for col in df.columns if col.startswith('sensor')]
    if scaler is None:
        scaler = MinMaxScaler()
        df[sensors_cols] = scaler.fit_transform(df[sensors_cols])
    else:
        df[sensors_cols] = scaler.transform(df[sensors_cols])

    return df, scaler

def build_official_test_set(scaler):
    """
    Processa o test_FD001 oficial e associa cada motor ao RUL_FD001.
    O protocolo NASA usa apenas o ultimo ciclo disponivel de cada motor de teste.
    """
    df_test = load_data(config.TEST_FILE)

    last_cycle_idx = df_test.groupby("engine_id")["time_cycle"].idxmax()
    df_last_cycles = df_test.loc[last_cycle_idx].sort_values("engine_id").reset_index(drop=True)

    rul_path = config.RAW_DATA_DIR / config.RUL_FILE
    y_official = pd.read_csv(rul_path, sep=r'\s+', header=None, names=["RUL"])
    y_official["RUL"] = y_official["RUL"].clip(upper=config.RUL_CAP)

    if len(df_last_cycles) != len(y_official):
        raise ValueError(
            "Quantidade de motores no test_FD001.txt difere das linhas em RUL_FD001.txt."
        )

    df_last_cycles["RUL"] = y_official["RUL"].to_numpy()
    df_processed_test, _ = process_data(df_last_cycles, scaler=scaler)
    return df_processed_test

def run_etl_pipeline():
    """
    Função principal que orquestra as etapas de Extração, 
    Transformação e Carga (ETL) dos dados.
    """

    # 1. Extração
    download_and_extract_data()
    # 2. Leitura
    df_raw = load_data(config.TRAIN_FILE)
    # 3. Transformação (Cálculo do alvo para a Regressão)
    df_processed = calculate_rul(df_raw)
    # 4. Processamento adicional (Remoção de sensores de baixa variância e normalização)
    df_processed, scaler = process_data(df_processed)
    # 5. Carga (Salvando o dado pronto para o treinamento)
    save_path = config.PROCESSED_DATA_DIR / config.PROCESSED_FILE_NAME
    df_processed.to_csv(save_path, index=False)

    df_official_test = build_official_test_set(scaler)
    official_test_path = config.PROCESSED_DATA_DIR / config.PROCESSED_OFFICIAL_TEST_FILE_NAME
    df_official_test.to_csv(official_test_path, index=False)
    
    print(f"Pipeline de dados concluído! Dataset processado salvo em: {save_path}")

    print(f"Teste oficial NASA salvo em: {official_test_path}")

    return scaler 

if __name__ == "__main__":
    # Teste do pipeline de dados de forma isolada
    run_etl_pipeline()
