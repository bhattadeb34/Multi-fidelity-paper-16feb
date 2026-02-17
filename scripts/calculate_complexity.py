
#!/usr/bin/env python3
"""
calculate_complexity.py
=======================
Calculate Böttcher Complexity Scores for the dataset using RDKit implementation.
"""

import sys
import os
import pandas as pd
import numpy as np
from rdkit import Chem

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DATA_DIR, NIST_EXP_PATH, OUT_DIR
from src.bottchscore import BottchScoreRDKit

OUTPUT_FILE = os.path.join(OUT_DIR, 'complexity_scores.csv')

def main():
    print('Loading dataset...')
    if not os.path.exists(NIST_EXP_PATH):
        print(f"Error: {NIST_EXP_PATH} not found.")
        return

    df = pd.read_csv(NIST_EXP_PATH)
    smiles_list = df['smiles'].values

    print(f'Calculating Böttcher Complexity for {len(smiles_list)} molecules...')
    scorer = BottchScoreRDKit(verbose=False)
    
    scores = []
    for i, smi in enumerate(smiles_list):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(smiles_list)}")
        try:
            mol = Chem.MolFromSmiles(str(smi))
            score = scorer.score(mol)
        except:
            score = 0.0
        scores.append(score)

    df['bottcher_complexity'] = scores
    
    print(f'Saving to {OUTPUT_FILE}...')
    df[['smiles', 'bottcher_complexity']].to_csv(OUTPUT_FILE, index=False)
    
    avg_score = np.mean(scores)
    print(f'Done. Average Complexity: {avg_score:.2f}')

if __name__ == '__main__':
    main()
