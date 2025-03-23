from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from collections import defaultdict

'''
This is a code to create a 2D representation of the pdb structure of the chignolin protein using rdkit.
The Tyr2 residue is highlighted, as it takes a key part in the stability of chignolin.
'''

pdb_path = "/Users/tsara/Desktop/mace/chig_draw/5awl_databank.pdb"
output_path = "/Users/tsara/Desktop/mace/chig_draw/chignolin_2d_highlight_Tyr2.png"
mol = Chem.MolFromPDBFile(pdb_path, removeHs=True)
AllChem.Compute2DCoords(mol)

# get atoms in Tyr2
tyr2_atoms = []
for atom in mol.GetAtoms():
    pdb_info = atom.GetPDBResidueInfo()
    if pdb_info and pdb_info.GetResidueName().strip() == "TYR" and pdb_info.GetResidueNumber() == 2:
        tyr2_atoms.append(atom.GetIdx())

# get bonds between Tyr2 atoms
highlight_bonds = []
for bond in mol.GetBonds():
    a1 = bond.GetBeginAtomIdx()
    a2 = bond.GetEndAtomIdx()
    if a1 in tyr2_atoms and a2 in tyr2_atoms:
        highlight_bonds.append(bond.GetIdx())

highlight_color = (1.0, 0.8, 0.2)
drawer = Draw.MolDraw2DCairo(1000, 1000)
drawer.DrawMolecule(
        mol,
        highlightAtoms=tyr2_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors={idx: highlight_color for idx in tyr2_atoms},
        highlightBondColors={idx: highlight_color for idx in highlight_bonds},
    )
drawer.FinishDrawing()
with open(output_path, "wb") as f:
    f.write(drawer.GetDrawingText())
