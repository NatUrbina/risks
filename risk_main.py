import boto3
import pandas as pd
from datetime import datetime

# Dataset Load
df = pd.read_csv("subsidy_requests.csv", parse_dates=['contract_date', 'request_date'])

# CLient creation Bedrock
bedrock = boto3.client("bedrock-runtime")

# Prompt definition
def build_prompt(client_id, contract_date, request_date, requested_amount):
    return f"""
Asistente antifraude, revisa esta solicitud de subvención:

Cliente: {client_id}
Fecha de firma del contrato: {contract_date.strftime('%Y-%m-%d')}
Fecha de solicitud de ayuda: {request_date.strftime('%Y-%m-%d')}
Importe solicitado: {requested_amount} EUR

La norma indica que no se pueden solicitar más de 50.000 EUR en un periodo de 1 año desde la firma del contrato. 
¿Esta solicitud es sospechosa de fraude? Responde solo con "Sí" o "No" y explica por qué brevemente.
"""

# CLaude call using prompt
def analyze_with_claude(prompt):
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0.2
        }),
        contentType="application/json"
    )
    result = json.loads(response['body'].read())
    return result['content'][0]['text'].strip()

# Dataset load
for _, row in df.iterrows():
    prompt = build_prompt(
        row["client_id"],
        row["contract_date"],
        row["request_date"],
        row["requested_amount"]
    )
    result = analyze_with_claude(prompt)
    print(f"[Cliente {row['client_id']}] Resultado: {result}")
