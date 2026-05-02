"""
Pipeline Principal: Orquestra as etapas de ETL, Treinamento, Avaliação, Explicabilidade e Assistente LLM
Este script é o ponto de entrada para o pipeline completo. Ele coordena a execução das diferentes
fases do projeto, garantindo que os dados sejam processados, o modelo seja treinado,
avaliado e que as explicações sejam geradas para o assistente LLM. 

O fluxo é organizado da seguinte forma:

1. Engenharia de Dados: Executa o pipeline de ETL para preparar os dados.
2. Torneio de Modelos (Treinamento e Avaliação): Itera sobre os algoritmos definidos
   no config.py, treinando-os e avaliando-os. Identifica o melhor modelo com base no R².
3. Explicabilidade (XAI) e Assistente LLM: Gera explicações exclusivamente para o
   modelo vencedor e traduz essas explicações para um formato compreensível para os operadores.
"""

# Imports
from src import config 
from src.data_pipeline import run_etl_pipeline
from src.evaluation import evaluate_model
from src.explainability import generate_explanations
from src.llm_assistant import generate_operator_report
from src.train import run_training_pipeline

def main():
    """
    Função principal que orquestra a execução de todo o pipeline.
    """  
    print("Iniciando o Pipeline da Indústria 5.0...")

    # Garante que a estrutura de diretórios existe
    config.create_directories()

    # 1. Engenharia de Dados:
    print("\n[Fase 1] Engenharia e Processamento de Dados")
    run_etl_pipeline()

    # Variáveis para rastrear o campeão do torneio
    best_r2 = -float('inf')
    campeao_nome = ""
    campeao_model = None
    campeao_run_id = None
    campeao_x_test = None

    print("\n[Fase 2 e 3] Iniciando o Torneio de Modelos Preditivos (RUL)")
    
    try:
        # Itera sobre o dicionário de modelos na configuração
        for nome_modelo, instancia_modelo in config.MODELS_TO_TRAIN.items():
            print(f"\n-> Avaliando: {nome_modelo}")
            
            # Treinamento
            model, x_test, y_test, run_id = run_training_pipeline(nome_modelo, instancia_modelo)
            
            # Avaliação Estatística
            mae, rmse, r2 = evaluate_model(model, x_test, y_test, run_id, nome_modelo)
            
            # Verificando se o modelo atual é o melhor
            if r2 > best_r2:
                best_r2 = r2
                campeao_nome = nome_modelo
                campeao_model = model
                campeao_run_id = run_id
                campeao_x_test = x_test

    except FileNotFoundError as e:
        print(f"Aviso: {e}")
        print("Pule o torneio por enquanto até o ETL estar totalmente integrado e os dados baixados.")
        return # Encerra o programa se não houver dados

    # Exibe o grande vencedor
    print("\n" + "="*50)
    print(f"Modelo Campeão: {campeao_nome} (R²: {best_r2:.4f})")
    print("="*50)

    # 4. Explicabilidade (XAI) e Assistente LLM (Apenas com o Campeão!):
    print(f"\n[Fase 4] XAI e Tradução para o Operador ({campeao_nome})")
    id_critico, rul, sensor_1, sensor_2 = generate_explanations(campeao_model, campeao_x_test, campeao_run_id)
    generate_operator_report(id_critico, rul, sensor_1, sensor_2)

    print("\nExecução do Pipeline Finalizada com Sucesso!")

if __name__ == "__main__":
    main()
