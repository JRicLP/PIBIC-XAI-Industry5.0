import os
import google.generativeai as genai

# Import das configurações globais
import src.config as config

def generate_operator_report(id_critico, rul_predito, sensor_1, sensor_2):

    """
    Recebe a saída matemática do XAI e utiliza a API do Gemini
    para gerar um laudo técnico acionável para o mecânico de pista.
    """
    print("\nConectando ao Assistente LLM...")

    # Buscando a chave da API nas variáveis de ambiente do sistema
    api_key = os.environ.get("GEMINI_API_KEY")

    # Camada de simulação para o caso da API Key não ser encontrada:
    if not api_key:
        print("Aviso: GEMINI_API_KEY não encontrada nas variáveis de ambiente.")
        print("Simulando a resposta do LLM para fins de teste de pipeline...\n")
        print("="*50)
        print(" Laudo do Assistente (Simulação)")
        print("="*50)
        print(f"Atenção Mecânico: O motor de índice {id_critico} apresenta falha iminente em aproximadamente {rul_predito:.0f} ciclos.")
        print("Inspecione imediatamente os componentes relacionados à {sensor_1} e verifique também a integridade da {sensor_2}.")
        print("="*50 + "\n")
        return

    # Configuração real da API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Engenharia de Prompt:
    prompt = f"""
    Você é um Engenheiro Aeronáutico Sênior auxiliando um mecânico de pista.
    Sua função é traduzir o diagnóstico de uma Inteligência Artificial para uma linguagem simples, direta e acionável.

    Dados da IA:
    - Máquina (Motor Index): {id_critico}
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
        response = model.generate_content(prompt)
        print("\n" + "="*50)
        print("Laudo do Assistente de IA:")
        print("="*50)
        print(response.text.strip())
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"[Error] Falha ao comunicar com a API do Gemini: {e}")

if __name__ == "__main__":
    print("Este script destina-se a ser orquestrado pelo main.py")
    