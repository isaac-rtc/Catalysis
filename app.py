from flask import Flask, render_template, jsonify, request
from openai import OpenAI
from ro5 import calculate_ro5
from tox_image import generate_toxicity_map
import os

app = Flask(__name__)

# -----------------------------
# OpenAI setup
# -----------------------------

OPENAI_ENABLED = bool(os.getenv("OPENAI_API_KEY"))

if not OPENAI_ENABLED:
    print("WARNING: OpenAI API key not set — using mock responses")


# -----------------------------
# Routes
# -----------------------------


@app.route("/", methods=["GET"])
def root():
    return render_template("base.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Toxicity similarity map + probability
    """
    data = request.get_json()
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing SMILES"}), 400

    result = generate_toxicity_map(data["smiles"])

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    LLM-based explanation (or mock)
    """
    data = request.get_json()
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing SMILES"}), 400

    smiles = data["smiles"]

    # Compute RO5
    ro5_results = calculate_ro5(smiles)
    if "error" in ro5_results:
        return jsonify({"error": ro5_results["error"]}), 400

    # Mock mode (safe for demo)
    if not OPENAI_ENABLED:
        return jsonify(
            {
                "text": (
                    "Mock analysis response.\n\n"
                    "• Lipinski assessment: uncertain\n"
                    "• Toxicity risk: uncertain\n"
                    "• Drug-likeness: uncertain\n\n"
                    "(OpenAI API key not configured)"
                ),
                "mock": True,
            }
        )

    # Build prompt
    prompt = f"""
You are a chemistry assistant.

Given the following molecule:
SMILES: {smiles}

Computed properties from RDKit:
- Molecular weight: {ro5_results['MW']}
- logP: {ro5_results['LogP']}
- Hydrogen bond donors: {ro5_results['HBD']}
- Hydrogen bond acceptors: {ro5_results['HBA']}
- Lipinski violations: {ro5_results['Violations']}

Tasks:
- Explain whether the molecule satisfies Lipinski’s Rule of Five
- Assess potential toxicity risk at a high level
- Summarize drug-likeness for a non-expert user

Guidelines:
- Keep under 200 words
- Avoid overly technical language
- Do NOT invent data
- If unsure, say "uncertain"
- Use bullet points
"""

    # OpenAI call (billing point)
    client = OpenAI()
    response = client.responses.create(model="gpt-5-nano", input=prompt)

    text_output = response.output_text.strip()

    return jsonify({"text": text_output})


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8017)
