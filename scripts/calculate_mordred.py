
#!/usr/bin/env python3
"""
calculate_mordred.py
====================
Script to calculate full range of Mordred descriptors (1,613 in 2D) for the dataset.
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR, NIST_EXP_PATH

OUTPUT_FILE = os.path.join(DATA_DIR, 'mordred_features_all.csv')

def calculate_mordred_batch(smiles_list):
    calc = Calculator(descriptors, ignore_3D=True)
    mols = []
    valid_indices = []
    for i, s in enumerate(smiles_list):
        try:
            m = Chem.MolFromSmiles(str(s))
            if m:
                mols.append(m)
                valid_indices.append(i)
            else:
                mols.append(None)
        except:
            mols.append(None)
    
    valid_mols = [m for m in mols if m is not None]
    if not valid_mols:
        return pd.DataFrame(), []

    print(f"  Calculating features for {len(valid_mols)} molecules (this may take a while)...")
    df_features = calc.pandas(valid_mols)
    
    # Clean: convert non-numeric to NaN
    df_features = df_features.apply(pd.to_numeric, errors='coerce')
    
    return df_features, valid_indices

def main():
    print('Loading dataset...')
    if not os.path.exists(NIST_EXP_PATH):
        print(f"Error: {NIST_EXP_PATH} not found.")
        return

    df = pd.read_csv(NIST_EXP_PATH)
    smiles = df['smiles'].values

    print(f'Calculating Mordred descriptors for {len(smiles)} molecules...')
    features, valid_indices = calculate_mordred_batch(smiles)

    # Re-align with original dataframe
    # Create empty dataframe with same index as df
    result_df = pd.DataFrame(index=df.index, columns=features.columns)
    
    # Fill in calculated values at valid indices
    # Map valid_indices (list index) to original df index
    original_indices = df.index[valid_indices]
    result_df.loc[original_indices] = features.values

    # Add canonical SMILES for alignment
    from src.utils import canon
    result_df['can_smiles'] = df['smiles'].astype(str).apply(canon)

    print(f'Saving to {OUTPUT_FILE}...')
    result_df.to_csv(OUTPUT_FILE, index=False)
    print('Done.')

if __name__ == '__main__':
    main()
