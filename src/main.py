# Imports

import src.config as config
from src.train import run_training_pipeline
from src.data_pipeline import run_etl_pipeline
from src.evaluation import evaluate_model
from src.explainability import generate_explanations
from src.llm_assistant import generate_operator_report

def main():
    
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
        model, X_test, y_test, run_id = run_training_pipeline()
    except FileNotFoundError as e:
        print(f"Aviso: {e}")
        print("Pule a Fase 2 por enquanto até o ETL estar integrado.")

    
    # 3. Avaliação Estatística:

    print("\n3 - Avaliação Estatística Rigorosa")
    evaluate_model(model, X_test, y_test, run_id)

    # 4. Explicabilidade (XAI) e Assistente LLM:
    
    print("\n4. XAI e Tradução para o Operador")
    id_critico, rul, sensor_1, sensor_2 = generate_explanations(model, X_test, run_id)
    generate_operator_report(id_critico, rul, sensor_1, sensor_2)

    print("Finalizado!")

if __name__ == "__main__":
    main()