from flask import Flask, render_template, jsonify, request
from openai import OpenAI
from ro5 import calculate_ro5
import os
import json

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI()

# Startup warning (safe, no return here)
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OpenAI API key not set — API calls disabled")


@app.route("/", methods=["GET"])
def root():
    return render_template("base.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    # Safety check: do not bill if key is missing
    if not os.getenv("OPENAI_API_KEY"):
        return (
            jsonify(
                {
                    "error": "OpenAI API key not configured",
                    "text": "Mock response (API key missing, OpenAI not called).",
                }
            ),
            400,
        )

    # For now, hardcode SMILES (UI input comes later)
    smiles = "CCO"

    # Compute RO5 using RDKit
    ro5_results = calculate_ro5(smiles)

    if isinstance(ro5_results, dict) and "error" in ro5_results:
        return jsonify({"error": ro5_results["error"]}), 400

    # Build prompt
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
- Explain whether the molecule satisfies Lipinski’s Rule of Five
- Assess potential toxicity risk at a high level
- Summarize drug-likeness for a non-expert user

Guidelines:
- Keep the response under 200 words
- Avoid overly technical language
- Do NOT invent data
- If unsure, say "uncertain"

Return a clear, readable paragraph response.
"""

    # Call OpenAI (THIS is the only billing point)
    response = client.responses.create(model="gpt-5-nano", input=prompt)

    text_output = response.output_text.strip()

    return jsonify({"text": text_output})


if __name__ == "__main__":
    app.run(debug=True)
