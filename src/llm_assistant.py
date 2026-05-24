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
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from google import genai

from src import config

def get_gemini_api_key():
    """
    Carrega o .env da raiz do projeto e retorna a chave do Gemini.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    return os.environ.get("GEMINI_API_KEY")

def extract_response_text(response):
    """
    Extrai texto da resposta do Gemini mesmo quando response.text vier vazio.
    """
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    parts_text = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else []
        for part in parts or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts_text.append(part_text)

    return "\n".join(parts_text).strip()

def print_empty_response_diagnostics(response):
    """
    Mostra informacoes uteis quando a API responde 200 OK, mas sem texto.
    """
    print("[Aviso] A API respondeu, mas nao retornou texto para o laudo.")

    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback:
        print(f"Prompt feedback: {prompt_feedback}")

    candidates = getattr(response, "candidates", None) or []
    for index, candidate in enumerate(candidates):
        finish_reason = getattr(candidate, "finish_reason", None)
        safety_ratings = getattr(candidate, "safety_ratings", None)
        print(f"Candidato {index}: finish_reason={finish_reason}")
        if safety_ratings:
            print(f"Candidato {index}: safety_ratings={safety_ratings}")

def save_operator_report(report_text, motor_critico, sensor_1, sensor_2, model_name=None, run_id=None):
    """
    Salva o laudo do operador em Markdown e registra no MLflow quando houver run_id.
    """
    if not report_text:
        return None

    engine_id = motor_critico.get("engine_id", "desconhecido") if isinstance(motor_critico, dict) else motor_critico
    time_cycle = motor_critico.get("time_cycle", "nao informado") if isinstance(motor_critico, dict) else "nao informado"
    rul_predito = motor_critico.get("rul_predito", "nao informado") if isinstance(motor_critico, dict) else "nao informado"
    model_label = model_name or "modelo_desconhecido"

    safe_model_name = str(model_label).replace(" ", "_")
    report_path = config.REPORTS_DIR / f"operator_report_{safe_model_name}_engine_{engine_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_md = f"""# Laudo do Assistente LLM

## Contexto

- Modelo campeao: {model_label}
- Engine ID: {engine_id}
- Ciclo operacional: {time_cycle}
- RUL predito: {rul_predito}
- Fator principal: {sensor_1}
- Fator secundario: {sensor_2}

## Laudo

{report_text}
"""
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Laudo do operador salvo em: {report_path}")

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(report_path), artifact_path="operator_reports")

    return report_path

def generate_operator_report(
    motor_critico,
    rul_predito=None,
    sensor_1=None,
    sensor_2=None,
    model_name=None,
    run_id=None
):

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
    api_key = get_gemini_api_key()

    # Camada de simulação para o caso da API Key não ser encontrada:
    if not api_key:
        print("Aviso: GEMINI_API_KEY nao encontrada nas variaveis de ambiente.")
        print("Simulando a resposta do LLM para fins de teste de pipeline...\n")
        report_text = (
            f"Atencao mecanico: O motor {engine_id} apresenta falha iminente "
            f"em aproximadamente {rul_predito:.0f} ciclos. "
            f"Inspecione imediatamente os componentes relacionados a {sensor_1} "
            f"e verifique tambem a integridade de {sensor_2}."
        )
        print("="*50)
        print(" Laudo do Assistente (Simulacao)")
        print("="*50)
        print(report_text)
        save_operator_report(report_text, motor_critico, sensor_1, sensor_2, model_name, run_id)
        return report_text

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
            model="gemini-2.5-flash",
            contents=prompt
        )
        report_text = extract_response_text(response)
        print("\n" + "="*50)
        print("Laudo do Assistente de IA:")
        print("="*50)
        if report_text:
            print(report_text)
            save_operator_report(report_text, motor_critico, sensor_1, sensor_2, model_name, run_id)
        else:
            print_empty_response_diagnostics(response)
        print("="*50 + "\n") 
        return report_text
        
    except Exception as e:
        print(f"[Error] Falha ao comunicar com a API do Gemini: {e}")
        return None

if __name__ == "__main__":
    print("Este script destina-se a ser orquestrado pelo main.py")
