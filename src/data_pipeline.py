import pandas as pd
import urllib.request
import zipfile
import os

# Importa as configurações globais
import src.config as config

def download_and_extract_data():

    """
    Baixa e extrai o dataset CMAPSS da NASA caso os arquivos brutos não existam no diretório local.
    """

    zip_path = config.RAW_DATA_DIR / "CMAPSSData.zip"
    
    # Verifica se o arquivo de treino já foi extraído
    train_file_path = config.RAW_DATA_DIR / config.TRAIN_FILE
    if train_file_path.exists():
        print("-> Dataset bruto já existente. Pulando download.")
        return

    print("Baixando dataset da NASA (CMAPSS)...")
    urllib.request.urlretrieve(config.DATASET_URL, zip_path)
    
    print("Extraindo arquivos...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(config.RAW_DATA_DIR)
        
    print("Download e extração concluídos.")

def load_data():

    """
    Carrega o arquivo de treino txt delimitado por espaços e aplica 
    as nomenclaturas das colunas definidas no config.py.
    """

    train_path = config.RAW_DATA_DIR / config.TRAIN_FILE
    print("Carregando dados brutos...")
    
    # O dataset original usa espaços como delimitadores e não possui cabeçalho
    df_train = pd.read_csv(train_path, sep=r'\s+', header=None, names=config.ALL_COLUMNS)
    return df_train

def calculate_rul(df):

    """
    Calcula a Vida Útil Restante (RUL) para cada motor com base no ciclo máximo.
    RUL = (Ciclo Máximo do Motor) - (Ciclo Atual)
    """

    print("Calculando o RUL (Remaining Useful Life)...")
    
    # Encontra o ciclo de vida máximo para cada motor
    max_cycles = df.groupby('engine_id')['time_cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    
    # Faz o merge com o dataset original
    df = df.merge(max_cycles, on=['engine_id'], how='left')
    
    # Calcula o RUL contínuo
    df['RUL'] = df['max_cycle'] - df['time_cycle']
    
    # Remove a coluna auxiliar 'max_cycle' para manter o dataset limpo
    df.drop('max_cycle', axis=1, inplace=True)
    
    return df

def run_etl_pipeline():

    """
    Função principal que orquestra as etapas de Extração, 
    Transformação e Carga (ETL) dos dados.
    """

    # 1. Extração
    download_and_extract_data()
    
    # 2. Leitura
    df_raw = load_data()
    
    # 3. Transformação (Cálculo do alvo para a Regressão)
    df_processed = calculate_rul(df_raw)
    
    # 4. Carga (Salvando o dado pronto para o treinamento)
    save_path = config.PROCESSED_DATA_DIR / config.PROCESSED_FILE_NAME
    df_processed.to_csv(save_path, index=False)
    
    print(f"Pipeline de dados concluído! Dataset processado salvo em: {save_path}")

if __name__ == "__main__":
    # Permite testar o pipeline de dados isoladamente
    run_etl_pipeline()