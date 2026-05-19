""" 
Arquivo: llm_assistant.py

Descrição: Este módulo é responsável por interagir com a API do Gemini para 
gerar laudos técnicos baseados na saída do modelo de XAI. Ele traduz os dados
técnicos em recomendações acionáveis para os mecânicos de pista.
"""

# Ponto de Melhoria: Implementar .env para utilizar a chave da API de forma mais 
# segura e prática, evitando a exposição acidental em código-fonte ou logs.

# Imports
import os
from google import genai

def generate_operator_report(motor_critico, rul_predito=None, sensor_1=None, sensor_2=None):

    """
    Recebe a saída matemática do XAI e utiliza a API do Gemini
    para gerar um laudo técnico acionável para o mecânico de pista.
    """
    print("\nConectando ao Assistente LLM...")

    if isinstance(motor_critico, dict):
        engine_id = motor_critico["engine_id"]
        time_cycle = motor_critico["time_cycle"]
        rul_predito = motor_critico["rul_predito"]
    else:
        engine_id = motor_critico
        time_cycle = None
    id_critico = engine_id

    # Buscando a chave da API nas variáveis de ambiente do sistema
    api_key = os.environ.get("GEMINI_API_KEY")

    # Camada de simulação para o caso da API Key não ser encontrada:
    if not api_key:
        print("Aviso: GEMINI_API_KEY não encontrada nas variáveis de ambiente.")
        print("Simulando a resposta do LLM para fins de teste de pipeline...\n")
        print("="*50)
        print(" Laudo do Assistente (Simulação)")
        print("="*50)
        print(
            f"Atenção Mecânico: O motor {engine_id} apresenta falha iminente "
            f"em aproximadamente {rul_predito:.0f} ciclos."
        )
        print(
            f"Inspecione imediatamente os componentes relacionados à {sensor_1}" 
            f" e verifique também a integridade da {sensor_2}."
        )
        return

    # Configuração real da API
    client = genai.Client(api_key=api_key)

    # Ponto de Melhoria: Adicionar tratamento de erros mais robusto para lidar com falhas de rede,
    # respostas inesperadas da API ou limites de taxa, garantindo que o sistema seja resiliente.

    # Ponto de Melhoria: Implementar caching de respostas para diagnósticos semelhantes,
    # reduzindo a latência e o custo de chamadas repetidas à API para casos comuns.

    # Engenharia de Prompt:
    prompt = f"""
    Você é um Engenheiro Aeronáutico Sênior auxiliando um mecânico de pista.
    Sua função é traduzir o diagnóstico de uma Inteligência Artificial para uma linguagem simples, direta e acionável.

    Dados da IA:
    - Motor: {engine_id}
    - Ciclo operacional: {time_cycle if time_cycle is not None else "não informado"}
    - Vida Útil Restante Estimada (RUL): {rul_predito:.1f} ciclos até a falha crítica.
    - Principal fator causando a degradação: Anomalia no {sensor_1}.
    - Fator secundário: Alteração no {sensor_2}.

    Tarefa:
    Escreva um laudo técnico curto (máximo de 3 frases) focado no mecânico. 
    Diga qual é a urgência e onde ele deve focar a inspeção física.
    Não use jargões de "Machine Learning" ou "IA", foque na física do motor.
    """

    try:
        print("Gerando laudo técnico com IA Generativa...")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        print("\n" + "="*50)
        print("Laudo do Assistente de IA:")
        print("="*50)
        print(response.text.strip())
        print("="*50 + "\n") 
        
    except Exception as e:
        print(f"[Error] Falha ao comunicar com a API do Gemini: {e}")

if __name__ == "__main__":
    print("Este script destina-se a ser orquestrado pelo main.py")
