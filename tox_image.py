import base64
import numpy as np
import xgboost as xgb
import matplotlib.cm as cm
from flask import Flask, request, jsonify, render_template
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


def load_xgb_model(path):
    # 1. Load the raw booster
    bst = xgb.Booster()
    bst.load_model(path)

    # 2. Create the classifier wrapper
    clf = xgb.XGBClassifier()
    clf._Booster = bst

    # 3. Manually initialize the label encoder to fix the 'classes_' error
    # This tells the wrapper how to map the 0 and 1 outputs
    le = LabelEncoder()
    le.classes_ = np.array([0, 1])
    clf._le = le

    clf.n_classes_ = 2
    return clf


model = load_xgb_model("tox_model.json")


def xgb_proba_wrapper(fingerprint):
    """
    Wrapper to convert RDKit fingerprint to format XGBoost expects.
    """
    fp_array = np.array(list(fingerprint)).reshape(1, -1)
    # Use the Booster directly if the wrapper still gives trouble
    # dmat = xgb.DMatrix(fp_array)
    # prob = model.get_booster().predict(dmat)[0]

    prob = model.predict_proba(fp_array)[:, 1][0]
    return float(prob)


@app.route("/")
def index():
    return render_template("base.html")


@app.route("/predict", methods=["POST"])
def predict():
    smiles = request.get_json()["smiles"]

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return jsonify({"error": "Invalid SMILES"}), 400

    try:
        # 1. Create the drawer
        drawer = Draw.MolDraw2DCairo(600, 600)

        # --- CRITICAL: Prepare the molecule coordinates ---
        # If the molecule doesn't have coordinates, the map will be empty
        Chem.rdDepictor.Compute2DCoords(mol)

        # 2. Generate the Similarity Map
        # We catch the return value (though we draw directly to drawer)
        SimilarityMaps.GetSimilarityMapForModel(
            mol,
            lambda m, i: SimilarityMaps.GetMorganFingerprint(
                m, atomId=i, radius=2, nBits=2048, fpType="bv"
            ),
            xgb_proba_wrapper,
            draw2d=drawer,
            colorMap=cm.PiYG_r,  # Reversed color mapping
        )

        # 3. Finalize the drawing
        drawer.FinishDrawing()

        # 4. Get the bytes and encode
        png_data = drawer.GetDrawingText()

        # Ensure it is bytes for base64
        if isinstance(png_data, str):
            png_data = png_data.encode("utf-8")

        b64_img = base64.b64encode(png_data).decode("utf-8")

        # 5. Get overall probability
        fp_overall = SimilarityMaps.GetMorganFingerprint(
            mol, atomId=-1, radius=2, nBits=2048, fpType="bv"
        )
        overall_prob = xgb_proba_wrapper(fp_overall)

        return jsonify({"toxicity_prob": overall_prob, "image_data": b64_img})

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


print("test")

if __name__ == "__main__":
    app.run(debug=True, port=8006)
