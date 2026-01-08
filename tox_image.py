import base64
from flask import Flask, request, jsonify, render_template
import xgboost as xgb
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps

app = Flask(__name__)

# --- Load Model Globally (Load once at startup) ---
# Ensure 'tox_model.json' is in the same folder as this script
model = xgb.XGBClassifier()
model.load_model("tox_model.json")


# --- Helper Functions ---
def xgb_proba_wrapper(fingerprint):
    """
    Wrapper to convert RDKit fingerprint to format XGBoost expects.
    """
    # Convert BitVector to numpy array
    fp_array = np.array(list(fingerprint)).reshape(1, -1)
    # Get probability of Class 1 (Toxic)
    prob = model.predict_proba(fp_array)[:, 1][0]
    return float(prob)


@app.route("/")
def index():
    # Serves the HTML file (see Step 2)
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    smiles = data.get("smiles")

    if not smiles:
        return jsonify({"error": "No SMILES provided"}), 400

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return jsonify({"error": "Invalid SMILES string"}), 400

    try:
        # 1. Setup RDKit Drawer (Canvas)
        # We use MolDraw2DCairo which creates a PNG in memory
        drawer = Draw.MolDraw2DCairo(600, 600)

        # 2. Generate the Similarity Map
        # Note: We use the same lambda and wrapper from your snippet
        SimilarityMaps.GetSimilarityMapForModel(
            mol,
            lambda m, i: SimilarityMaps.GetMorganFingerprint(
                m, atomId=i, radius=2, nBits=2048, fpType="bv"
            ),
            xgb_proba_wrapper,
            draw2d=drawer,
        )

        # 3. Finish Drawing and Get Binary Data
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()

        # 4. Convert Binary PNG to Base64 String
        # This allows us to send the image as text over JSON
        b64_img = base64.b64encode(png_data).decode("utf-8")

        # 5. Get overall probability for display
        # (We calculate this separately just to show the number on the UI)
        fp_overall = SimilarityMaps.GetMorganFingerprint(
            mol, atomId=-1, radius=2, nBits=2048, fpType="bv"
        )
        overall_prob = xgb_proba_wrapper(fp_overall)

        return jsonify({"toxicity_prob": overall_prob, "image_data": b64_img})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
