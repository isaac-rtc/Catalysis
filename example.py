from openai import OpenAI
from ro5 import calculate_ro5
import json

client = OpenAI()

smiles = "CCO"

ro5_results = calculate_ro5(smiles)

if "error" in ro5_results:
    raise ValueError(ro5_results["error"])

prompt = f"""
You are a chemistry assistant.

Given the following molecule:
SMILES: {smiles}

Computed properties from RDKit:
- Molecular weight: {ro5_results["MW"]}
- logP: {ro5_results["LogP"]}
- Hydrogen bond donors: {ro5_results["HBD"]}
- Hydrogen bond acceptors: {ro5_results["HBA"]}
- Lipinski violations: {ro5_results["Violations"]}

Tasks:
1. Explain whether the molecule satisfies Lipinski’s Rule of Four.
2. Assess potential toxicity risk at a high level.
3. Summarize drug-likeness for a non-expert user.
4. Keep word count within 200, do not get too scientific.

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
print(raw_output)

cleaned = raw_output.strip()

if cleaned.startswith("```"):
    cleaned = cleaned.split("```")[1]

parsed = json.loads(cleaned)
print(parsed)
