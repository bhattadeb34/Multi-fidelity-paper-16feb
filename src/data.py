
import pandas as pd
import numpy as np
from typing import Tuple, List
from .config import NIST_NEU_PATH, NIST_PROT_PATH, NIST_EXP_PATH
from .utils import canon

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and merge all data sources."""
    print('Loading experimental and PM7 data...')

    d_neu = pd.read_csv(NIST_NEU_PATH)
    d_prot = pd.read_csv(NIST_PROT_PATH)
    d_exp = pd.read_csv(NIST_EXP_PATH)

    # Canonicalize SMILES
    d_neu['can'] = d_neu['smiles'].apply(canon)
    d_prot['can'] = d_prot['neutral_smiles'].apply(canon)
    d_exp['can'] = d_exp['smiles'].astype(str).apply(canon)

    # Select most stable protonated form (lowest HOF)
    d_prot_best = d_prot.sort_values(['can', 'heat_of_formation']).drop_duplicates('can')

    # Merge datasets
    m = pd.merge(d_neu, d_prot_best, on='can', suffixes=('_n', '_p'))
    df = pd.merge(m, d_exp, on='can')

    # Extract experimental PA (convert kJ to kcal if needed)
    y_exp = df['EXP_PA'].values.astype(float)
    if y_exp.mean() > 500:
        y_exp = y_exp / 4.184

    # Calculate PM7 PA baseline: HOF(neutral) + HOF(H+) - HOF(protonated)
    hof_n = df['heat_of_formation_n'].values.astype(float)
    hof_p = df['heat_of_formation_p'].values.astype(float)
    y_pm7 = hof_n + 365.7 - hof_p  # H+ HOF = 365.7 kcal/mol

    print(f'  Loaded {len(df)} molecules')
    print(f'  Experimental PA range: {y_exp.min():.1f} - {y_exp.max():.1f} kcal/mol')
    print(f'  PM7 PA range: {y_pm7.min():.1f} - {y_pm7.max():.1f} kcal/mol')

    return df, d_exp, y_exp, y_pm7
