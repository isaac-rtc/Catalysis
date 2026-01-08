from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def calculate_ro5(smiles: str) -> dict:
    """
    Calculate Lipinski's Rule of Five parameters for a given SMILES string.
    Returns a dictionary with molecular weight, LogP, number of H-bond donors,
    number of H-bond acceptors, and number of violations.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"error": "Invalid SMILES"}

    results = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
    }

    # Check violations
    violations = 0
    if results["MW"] >= 500:
        violations += 1
    if results["LogP"] >= 5:
        violations += 1
    if results["HBD"] > 5:
        violations += 1
    if results["HBA"] > 10:
        violations += 1

    results["Violations"] = violations
    return results


# Example: Caffeine
# print(calculate_ro5("CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))
