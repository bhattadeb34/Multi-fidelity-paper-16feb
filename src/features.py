
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys
from collections import Counter
from .config import BASE_DIR, DATA_DIR
from .utils import safe_float, canon

def compute_mordred_features(d_exp: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Load ALL Mordred 2D descriptors from pre-computed file (1521 descriptors).
    Falls back to the ~100 descriptors in Dataset.csv if the file is missing.
    """
    mordred_file = os.path.join(DATA_DIR, 'mordred_features_all.csv')

    if os.path.exists(mordred_file):
        print(f'  Loading pre-computed Mordred features from mordred_features_all.csv...')
        df_mord = pd.read_csv(mordred_file)
        mordred_cols = [c for c in df_mord.columns if c != 'can_smiles']
        # Store for alignment
        compute_mordred_features._df_mord = df_mord.copy()
        
        # We don't return the full matrix here because alignment happens later
        # Just return dummy for now or full aligned?
        # The main logic uses _df_mord attribute for alignment later.
        # But it returns X_mordred which is seemingly unused if alignment logic runs?
        # Wait, the main logic:
        # X_mord, names_mord = compute_mordred_features(d_exp)
        # ... usage of _df_mord ...
        # If _df_mord is None, it uses X_mord directly (if indices match d_exp directly?)
        # Let's keep original logic structure.
        
        X_mordred = df_mord[mordred_cols].values.astype(float)
        X_mordred = np.nan_to_num(X_mordred)
        names = [f'Mordred_{c}' for c in mordred_cols]
        print(f'    Loaded {len(names)} Mordred features')
        return X_mordred, names
    else:
        print(f'  WARNING: mordred_features_all.csv not found, falling back to Dataset.csv')
        exclude_patterns = ['fp_', 'smiles', 'reg_num', 'EXP_PA', 'Ehomo', 'Elumo', 'chemical potential',
                            'hardness', 'MK_charge', 'Dipole moment', 'CM5_charge']
        mordred_cols = []
        for col in d_exp.columns:
            is_excluded = any(pat in col for pat in exclude_patterns)
            if not is_excluded:
                if d_exp[col].dtype in [np.float64, np.int64, float, int]:
                    mordred_cols.append(col)
        X_mordred = d_exp[mordred_cols].values.astype(float)
        X_mordred = np.nan_to_num(X_mordred)
        names = [f'Mordred_{c}' for c in mordred_cols]
        print(f'    Extracted {len(names)} Mordred features (fallback)')
        compute_mordred_features._df_mord = None
        return X_mordred, names


def compute_maccs_fingerprints(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Compute MACCS fingerprints (167 bits)."""
    print('  Computing MACCS fingerprints (167 bits)...')
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros(167, dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        else:
            fps.append(np.zeros(167, dtype=int))
    X_maccs = np.array(fps)
    names = [f'MACCS_{i}' for i in range(167)]
    print(f'    Generated {X_maccs.shape[1]} MACCS bits')
    return X_maccs, names


def compute_morgan_fingerprints(smiles_list: List[str], radius: int = 2, n_bits: int = 1024) -> Tuple[np.ndarray, List[str]]:
    """Compute Morgan/ECFP fingerprints."""
    print(f'  Computing Morgan fingerprints (r={radius}, {n_bits} bits)...')
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        else:
            fps.append(np.zeros(n_bits, dtype=int))
    X_morgan = np.array(fps)
    names = [f'Morgan_{i}' for i in range(n_bits)]
    print(f'    Generated {X_morgan.shape[1]} Morgan bits')
    return X_morgan, names


def rdkit_descriptor_names() -> List[str]:
    """Get RDKit 2D descriptor names."""
    return [name for name, _ in Descriptors._descList]


def rdkit_descriptor_values(mol) -> List[float]:
    """Calculate RDKit 2D descriptors."""
    vals = []
    for _, func in Descriptors._descList:
        try:
            val = func(mol)
            vals.append(safe_float(val))
        except Exception:
            vals.append(0.0)
    return vals


def compute_rdkit_descriptors_delta(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Compute RDKit descriptors for neutral, protonated, and delta."""
    print('  Computing RDKit descriptors (neutral, protonated, delta)...')
    names_2d = rdkit_descriptor_names()
    vals_neu, vals_prot = [], []

    smiles_n = df['smiles_n'] if 'smiles_n' in df.columns else df['can']
    smiles_p = df['smiles_p'] if 'smiles_p' in df.columns else df['can']

    for i, (s_n, s_p) in enumerate(zip(smiles_n, smiles_p)):
        if i % 200 == 0:
            print(f'    Processing molecule {i}/{len(df)}...')
        mol_n = Chem.MolFromSmiles(str(s_n))
        mol_p = Chem.MolFromSmiles(str(s_p))
        vals_neu.append(rdkit_descriptor_values(mol_n) if mol_n else [0.0]*len(names_2d))
        vals_prot.append(rdkit_descriptor_values(mol_p) if mol_p else [0.0]*len(names_2d))

    neu_arr = np.array(vals_neu)
    prot_arr = np.array(vals_prot)
    delta_arr = prot_arr - neu_arr
    X = np.hstack([neu_arr, prot_arr, delta_arr])
    X = np.nan_to_num(X)
    names = ([f'RDKit_{n}_Neu' for n in names_2d] +
             [f'RDKit_{n}_Prot' for n in names_2d] +
             [f'RDKit_{n}_Delta' for n in names_2d])
    print(f'    Generated {len(names)} RDKit features ({len(names_2d)} x 3)')
    return X, names


def compute_pm7_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Extract PM7 quantum features."""
    print('  Extracting PM7 quantum features...')
    pm7_cols = ['heat_of_formation', 'dipole_x', 'dipole_y', 'dipole_z', 'dipole_moment',
                'homo_ev', 'lumo_ev', 'gap_ev', 'ionization_potential',
                'cosmo_area', 'cosmo_volume', 'total_energy_kcal_mol', 'num_atoms']
    rows = []
    for _, row in df.iterrows():
        feat = []
        for col in pm7_cols:
            col_n = f'{col}_n'
            col_p = f'{col}_p'
            val_n = safe_float(row.get(col_n, 0.0))
            val_p = safe_float(row.get(col_p, 0.0))
            feat.extend([val_n, val_p, val_p - val_n])
        rows.append(feat)
    X_pm7 = np.array(rows)
    X_pm7 = np.nan_to_num(X_pm7)
    names = []
    for col in pm7_cols:
        names.extend([f'PM7_{col}_Neu', f'PM7_{col}_Prot', f'PM7_{col}_Delta'])
    print(f'    Generated {len(names)} PM7 features')
    return X_pm7, names


def compute_3d_descriptors(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Compute 3D descriptors."""
    print('  Computing 3D descriptors (conformer-based)...')
    names_3d = ['RadGyration', 'PMI1', 'PMI2', 'PMI3', 'Spherocity',
                'Asphericity', 'Eccentricity', 'InertialShapeFactor', 'NPR1', 'NPR2']
    rows = []
    for i, smi in enumerate(smiles_list):
        if i % 200 == 0:
            print(f'    Processing molecule {i}/{len(smiles_list)}...')
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            try:
                m = Chem.AddHs(mol)
                AllChem.EmbedMolecule(m, AllChem.ETKDGv2())
                AllChem.MMFFOptimizeMolecule(m)
                vals = [rdMolDescriptors.CalcRadiusOfGyration(m),
                        rdMolDescriptors.CalcPMI1(m), rdMolDescriptors.CalcPMI2(m), rdMolDescriptors.CalcPMI3(m),
                        rdMolDescriptors.CalcSpherocityIndex(m), rdMolDescriptors.CalcAsphericity(m),
                        rdMolDescriptors.CalcEccentricity(m), rdMolDescriptors.CalcInertialShapeFactor(m),
                        rdMolDescriptors.CalcNPR1(m), rdMolDescriptors.CalcNPR2(m)]
                rows.append([safe_float(v) for v in vals])
            except Exception:
                rows.append([0.0] * len(names_3d))
        else:
            rows.append([0.0] * len(names_3d))
    X_3d = np.array(rows)
    names = [f'3D_{n}' for n in names_3d]
    print(f'    Generated {len(names)} 3D features')
    return X_3d, names


def build_feature_matrix(df: pd.DataFrame, d_exp: pd.DataFrame, compute_3d: bool = True, recompute: bool = False):
    """Build comprehensive feature matrix."""
    print('\n' + '=' * 60)
    print('FEATURE ENGINEERING')
    print('=' * 60)
    
    # Check cache
    cache_file = os.path.join(DATA_DIR, 'feature_matrix_full.pkl')
    if not recompute and os.path.exists(cache_file):
        print(f'Loading cached features from {cache_file}...')
        try:
             # Load pickle
             import pickle
             with open(cache_file, 'rb') as f:
                 data = pickle.load(f)
             print('  Successfully loaded cached features.')
             return data['X'], data['names'], data['sources'], data['counts']
        except Exception as e:
            print(f'  Error loading cache: {e}. Recomputing...')

    all_X = []
    all_names = []
    all_sources = []
    smiles = df['can'].values

    # 1. Mordred
    X_mord, names_mord = compute_mordred_features(d_exp)
    df_mord = getattr(compute_mordred_features, '_df_mord', None)
    if df_mord is not None and '_can' in df_mord.columns:
        mord_aligned = df_mord.set_index('_can')
        mord_cols = [c for c in mord_aligned.columns if c != 'can_smiles']
        X_mord_aligned = []
        for s in smiles:
            if s in mord_aligned.index:
                # Handle duplicates if any, take first
                if isinstance(mord_aligned.loc[s], pd.DataFrame):
                     vals = mord_aligned.loc[s].iloc[0][mord_cols].values.astype(float)
                else:
                     vals = mord_aligned.loc[s][mord_cols].values.astype(float)
                X_mord_aligned.append(vals)
            else:
                X_mord_aligned.append(np.zeros(len(mord_cols)))
        X_mord = np.array(X_mord_aligned)
    else:
        # Fallback alignment via d_exp index logic (assuming d_exp order matches if simplistic)
        # But wait, original code logic was confusing there.
        # Original: "mord_aligned = d_exp.set_index(d_exp['smiles'].apply(canon))" "mord_cols = None"
        # "X_mord_aligned.append(X_mord[idx])"
        # This implies X_mord was generated from d_exp IN ORDER.
        # But if we filter smiles, indices might shift?
        # The key is `df['can']` vs `d_exp['can']`.
        # `df` is merge of `d_exp` and `d_neu`. `d_exp` order != `df` order necessarily.
        # So we MUST align.
        # Original lines 364-375 handle alignment if falling back.
        # Let's replicate strict logic.
        mord_aligned = pd.DataFrame(X_mord, index=d_exp['can'])
        X_mord_aligned = []
        for s in smiles:
            if s in mord_aligned.index:
                 vals = mord_aligned.loc[s].values
                 if vals.ndim > 1: vals = vals[0]
                 X_mord_aligned.append(vals)
            else:
                X_mord_aligned.append(np.zeros(X_mord.shape[1]))
        X_mord = np.array(X_mord_aligned)

    all_X.append(X_mord)
    all_names.extend(names_mord)
    all_sources.extend(['Mordred'] * len(names_mord))

    # 2. MACCS
    X_maccs, names_maccs = compute_maccs_fingerprints(smiles)
    all_X.append(X_maccs)
    all_names.extend(names_maccs)
    all_sources.extend(['MACCS'] * len(names_maccs))

    # 3. Morgan
    X_morgan, names_morgan = compute_morgan_fingerprints(smiles)
    all_X.append(X_morgan)
    all_names.extend(names_morgan)
    all_sources.extend(['Morgan'] * len(names_morgan))

    # 4. RDKit
    X_rdkit, names_rdkit = compute_rdkit_descriptors_delta(df)
    all_X.append(X_rdkit)
    all_names.extend(names_rdkit)
    all_sources.extend(['RDKit'] * len(names_rdkit))

    # 5. PM7
    X_pm7, names_pm7 = compute_pm7_features(df)
    all_X.append(X_pm7)
    all_names.extend(names_pm7)
    all_sources.extend(['PM7'] * len(names_pm7))

    # 6. 3D
    if compute_3d:
        X_3d, names_3d = compute_3d_descriptors(smiles)
        all_X.append(X_3d)
        all_names.extend(names_3d)
        all_sources.extend(['3D'] * len(names_3d))

    X_all = np.hstack(all_X)
    X_all = np.nan_to_num(X_all)

    print('\nFeature Summary:')
    initial_counts = Counter(all_sources)
    for source, count in initial_counts.items():
        print(f'  {source}: {count} features')
    print(f'  TOTAL: {len(all_names)} features')

    print(f'  TOTAL: {len(all_names)} features')

    # Save cache
    print(f'Saving features to {cache_file}...')
    import pickle
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'X': X_all,
            'names': all_names,
            'sources': all_sources,
            'counts': dict(initial_counts)
        }, f)

    return X_all, all_names, all_sources, dict(initial_counts)
