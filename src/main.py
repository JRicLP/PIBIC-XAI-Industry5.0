"""
Pipeline Principal: Orquestra as etapas de ETL, Treinamento, Avaliação, Explicabilidade e Assistente LLM
Este script é o ponto de entrada para o pipeline completo. Ele coordena a execução das diferentes
fases do projeto, garantindo que os dados sejam processados, o modelo seja treinado,
avaliado e que as explicações sejam geradas para o assistente LLM. 

O fluxo é organizado da seguinte forma:

1. Engenharia de Dados: Executa o pipeline de ETL para preparar os dados.
2. Treinamento e Governança: Treina o modelo preditivo de RUL e registra os resultados no MLflow.
3. Avaliação Estatística: Avalia o desempenho do modelo usando métricas rigorosas.
4. Explicabilidade (XAI) e Assistente LLM: Gera explicações para as previsões do modelo e 
traduz essas explicações para um formato compreensível para os operadores.

Cada etapa é modularizada em arquivos separados para manter a
organização e facilitar a manutenção do código.
"""

# Imports
from src import config # Modificado para importar o módulo de configuração global
from src.data_pipeline import run_etl_pipeline
from src.evaluation import evaluate_model
from src.explainability import generate_explanations
from src.llm_assistant import generate_operator_report
from src.train import run_training_pipeline

def main():

    """
    Função principal que orquestra a execução de todo o pipeline. 
    Ela garante que cada etapa seja executada na ordem correta e lida com quaisquer dependências
    entre as fases.
    O resultado final é um modelo treinado, avaliado e explicações geradas para o assistente LLM.
    """  
    print("Iniciando o Pipeline...")

    # Garante que a estrutura de diretórios existe
    config.create_directories()

    # 1. Engenharia de Dados:
    print("\n1 - Engenharia e Processamento de Dados")
    run_etl_pipeline()
    print("-> (Pendente: Módulo data_pipeline.py)")
    # 2. Treinamento e Governança:
    print("\n2 - Treinamento do Modelo Preditivo (RUL)")
    try:
        model, x_test, y_test, run_id = run_training_pipeline()
    except FileNotFoundError as e:
        print(f"Aviso: {e}")
        print("Pule a Fase 2 por enquanto até o ETL estar integrado.")
    # 3. Avaliação Estatística:
    print("\n3 - Avaliação Estatística Rigorosa")
    evaluate_model(model, x_test, y_test, run_id)
    # 4. Explicabilidade (XAI) e Assistente LLM:
    print("\n4. XAI e Tradução para o Operador")
    id_critico, rul, sensor_1, sensor_2 = generate_explanations(model, x_test, run_id)
    generate_operator_report(id_critico, rul, sensor_1, sensor_2)

    print("Finalizado!")

if __name__ == "__main__":
    main()
