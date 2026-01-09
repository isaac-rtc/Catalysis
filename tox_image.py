import base64
import numpy as np
import xgboost as xgb
import matplotlib.cm as cm
from rdkit import Chem
import rdkit.Chem.Draw as Draw
from rdkit.Chem.Draw import SimilarityMaps
from sklearn.preprocessing import LabelEncoder


# -------------------------------
# Model loading (unchanged logic)
# -------------------------------


def load_xgb_model(path):
    bst = xgb.Booster()
    bst.load_model(path)

    clf = xgb.XGBClassifier()
    clf._Booster = bst

    le = LabelEncoder()
    le.classes_ = np.array([0, 1])
    clf._le = le
    clf.n_classes_ = 2

    return clf


model = load_xgb_model("tox_model.json")


def xgb_proba_wrapper(fingerprint):
    fp_array = np.array(list(fingerprint)).reshape(1, -1)
    prob = model.predict_proba(fp_array)[:, 1][0]
    return float(prob)


# ----------------------------------------
# MAIN FUNCTION CALLED BY FLASK (IMPORTANT)
# ----------------------------------------


def generate_toxicity_map(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"error": "Invalid SMILES"}

    try:
        # Prepare drawing
        drawer = Draw.MolDraw2DCairo(600, 600)
        Chem.rdDepictor.Compute2DCoords(mol)

        # Generate similarity map
        SimilarityMaps.GetSimilarityMapForModel(
            mol,
            lambda m, i: SimilarityMaps.GetMorganFingerprint(
                m, atomId=i, radius=2, nBits=2048, fpType="bv"
            ),
            xgb_proba_wrapper,
            draw2d=drawer,
            colorMap=cm.PiYG_r,
        )

        drawer.FinishDrawing()

        # Encode image
        png_data = drawer.GetDrawingText()
        if isinstance(png_data, str):
            png_data = png_data.encode("utf-8")

        b64_img = base64.b64encode(png_data).decode("utf-8")

        # Overall probability
        fp_overall = SimilarityMaps.GetMorganFingerprint(
            mol, atomId=-1, radius=2, nBits=2048, fpType="bv"
        )
        overall_prob = xgb_proba_wrapper(fp_overall)

        return {"toxicity_prob": overall_prob, "image_data": b64_img}

    except Exception as e:
        return {"error": str(e)}
