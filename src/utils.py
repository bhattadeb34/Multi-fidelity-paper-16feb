
import math
import numpy as np
import pandas as pd
from rdkit import Chem

def canon(smiles: str) -> str:
    """Canonicalize SMILES string."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return str(smiles)
    except Exception:
        return str(smiles)

def safe_float(x, default: float = 0.0) -> float:
    """Safely convert to float."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default
