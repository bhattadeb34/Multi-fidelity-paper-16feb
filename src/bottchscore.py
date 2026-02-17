
#!/usr/bin/env python3
"""
RDKit implementation of the Böttcher Complexity Score.
Ported from bottchscore3.py to replicate logic without OpenBabel.
"""

import sys
from math import log
from rdkit import Chem
# from rdkit.Chem import AllChem # Unused

class BottchScoreRDKit:
    def __init__(self, verbose=False):
        self.verbose = verbose
        # Ported from bottchscore3.py
        self._mesomery_patterns = {
            # SMARTS_pattern : [contribution]
            '[$([#8;X1])]=*-[$([#8;X1])]': {'indices': [0, 2], 'contribution': 1.5},  
            '[$([#7;X2](=*))](=*)(-*=*)': {'indices': [2, 1], 'contribution': 1.5}, 
        }

    def score(self, mol):
        if not mol: return 0.0
        
        try:
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        except:
            pass

        # 1. Mesomeric Corrections
        mesomery_updates = {}
        for smarts, info in self._mesomery_patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if not pattern: continue
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                target_indices = info['indices']
                contrib = info['contribution']
                for t_idx in target_indices:
                    atom_idx = match[t_idx]
                    mesomery_updates[atom_idx] = contrib

        # 2. Symmetry (d_i)
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        
        atom_data = {} # idx -> {di, ei, si, Vi, bi, complexity}
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if atom.GetAtomicNum() == 1: continue
            
            # d_i: Non-equivalent neighbors
            neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
            neigh_ranks = {ranks[n.GetIdx()] for n in neighbors}
            d_i = len(neigh_ranks)
            
            # e_i: Unique elements (including self)
            elems = {atom.GetAtomicNum()}
            for n in neighbors: elems.add(n.GetAtomicNum())
            e_i = len(elems)
            
            # s_i: Stereochemistry (Base)
            s_i = 1
            if atom.GetChiralTag() != Chem.CHI_UNSPECIFIED:
                s_i += 1
            
            # V_i: Valence (TotalValence usually correct in RDKit)
            V_i = 8 - atom.GetTotalValence() + atom.GetFormalCharge()
            
            # b_i: Bond Orders
            b_i = 0.0
            for bond in atom.GetBonds():
                nbr = bond.GetOtherAtom(atom)
                if nbr.GetAtomicNum() == 1: continue
                
                if idx in mesomery_updates:
                    contribution = mesomery_updates[idx]
                else:
                    bt = bond.GetBondType()
                    if bt == Chem.BondType.DOUBLE: contribution = 2.0
                    elif bt == Chem.BondType.TRIPLE: contribution = 3.0
                    elif bt == Chem.BondType.AROMATIC: contribution = 1.5
                    else: contribution = 1.0
                
                b_i += contribution
            
            # Local Complexity
            try:
                if V_i <= 0 or b_i <= 0:
                    local_c = 0
                else:
                    local_c = d_i * e_i * s_i * log(V_i * b_i, 2)
            except:
                local_c = 0
                
            atom_data[idx] = {'c': local_c, 'si': s_i, 'di':d_i, 'ei':e_i, 'Vi':V_i, 'bi':b_i}

        # 3. E/Z Isomer Logic
        ez_atoms = set()
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                stereo = bond.GetStereo()
                if stereo in [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ]:
                    a1 = bond.GetBeginAtom().GetIdx()
                    a2 = bond.GetEndAtom().GetIdx()
                    
                    if a1 in atom_data and a2 in atom_data:
                        ez_atoms.add(a1)
                        ez_atoms.add(a2)
                        
                        c1 = atom_data[a1]['c']
                        c2 = atom_data[a2]['c']
                        
                        if c1 <= c2:
                            atom_data[a1]['c'] *= 2
                            atom_data[a1]['si'] += 1
                        else:
                            atom_data[a2]['c'] *= 2
                            atom_data[a2]['si'] += 1

        # 4. Total Complexity Sum
        total_complexity = sum(d['c'] for d in atom_data.values())
        
        # 5. Symmetry Correction
        rank_groups = {}
        for idx in atom_data:
            r = ranks[idx]
            if r not in rank_groups: rank_groups[r] = []
            rank_groups[r].append(idx)
            
        correction = 0.0
        for r, group in rank_groups.items():
            if len(group) > 1:
                for idx in group:
                    correction += 0.5 * atom_data[idx]['c']
        
        if self.verbose:
            self.print_table(atom_data, ranks, mol)
                
        return total_complexity - correction

    def print_table(self, atom_data, ranks, mol):
        print("\nDEBUG TABLE")
        print("Idx\tSym\tdi\tei\tsi\tVi\tbi\tComplex")
        for idx in sorted(atom_data.keys()):
            d = atom_data[idx]
            sym = mol.GetAtomWithIdx(idx).GetSymbol()
            print(f"{idx}\t{sym}\t{d['di']}\t{d['ei']}\t{d['si']}\t{d['Vi']}\t{d['bi']:.1f}\t{d['c']:.2f}")
        print("------------------------------------------------")

def calculate_bottchscore(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        scorer = BottchScoreRDKit()
        return scorer.score(mol)
    except:
        return 0.0
