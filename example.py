from openai import OpenAI
import json

client = OpenAI()

smiles = "CCO"

lipinski_results = {
    "molecular_weight": 180.16,
    "logP": 1.2,
    "hbd": 1,
    "hba": 2,
    "violations": 0,
}


prompt = f"""
You are a chemistry assistant.

Given the following molecule:
SMILES: {smiles}

Computed properties from RDKit:
- Molecular weight: {lipinski_results["molecular_weight"]}
- logP: {lipinski_results["logP"]}
- Hydrogen bond donors: {lipinski_results["hbd"]}
- Hydrogen bond acceptors: {lipinski_results["hba"]}
- Lipinski violations: {lipinski_results["violations"]}

Tasks:
1. Explain whether the molecule satisfies Lipinski’s Rule of Five.
2. Assess potential toxicity risk at a high level.
3. Summarize drug-likeness for a non-expert user.

Return your response strictly as valid JSON with the following keys:
{{
  "lipinski_explanation": "string",
  "toxicity_risk": "low | moderate | high | uncertain",
  "toxicity_reasoning": "string",
  "drug_likeness_summary": "string",
  "confidence_notes": "string"
}}

Do NOT calculate new numerical values.
Do NOT invent data.
If unsure, say "uncertain".
"""


response = client.responses.create(model="gpt-5-nano", input=prompt)

raw_output = response.output_text
print("RAW OUTPUT:")
print(raw_output)


cleaned = raw_output.strip()

if cleaned.startswith("```"):
    cleaned = cleaned.split("```")[1]

parsed = json.loads(cleaned)

print("\nPARSED OUTPUT:")
print(parsed)
