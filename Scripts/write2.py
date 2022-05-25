from pysmiles import read_smiles
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from Bio.PDB import PDBParser
import pdb
from biopandas.mol2 import PandasMol2

folders = os.listdir('data/v2015/')
home = os.getcwd()
rows = []
elements = {}
allowed = ['N', 'O', 'S', 'F', 'Cl', 'P', 'H', 'C']

with open('data/pdbbind_filter.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    for folder in folders:
        sloppyparser = PDBParser(PERMISSIVE=True)
        try:
            protein = sloppyparser.get_structure("MD_system", 'data/v2015/' + folder + '/' + folder + '_protein.pdb')
            ligand = PandasMol2().read_mol2('data/v2015/' + folder + '/' + folder + '_ligand.mol2')
        except:
            continue
        pr_atoms = []
        li_atoms = []
        write_mol = True

        for atom in protein.get_atoms():
            pr_atoms.append(atom)

            if atom.element not in allowed:
                print(atom.element)
                write_mol = False
                break

        for atom in ligand.df['atom_type']:
            atom_ele = atom.split('.')[0]
            li_atoms.append(atom_ele)

            if atom_ele not in allowed:
                print("Ligand error\t", folder)
                print(atom_ele)
                write_mol = False
                break

        if not write_mol:
            continue

        row = [folder, len(pr_atoms), len(li_atoms)]
        # rows.append(row)
        writer.writerow(row)
