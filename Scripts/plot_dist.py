import pdb
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

file = 'data/pdbbind_filter.csv'
protein = []
ligand = []
with open(file, 'r') as file:
    filecontent = csv.reader(file)
    for row in filecontent:
        protein.append(float(row[1]))
        ligand.append(float(row[2]))

x_min = 0
x_max = 30000

z = {}

z['Protein Size'] = np.array(protein)
z['Ligand Size'] = np.array(ligand)

fig, axes = plt.subplots(2, 1)

plot = sns.histplot(
    data=protein, kde=True, bins=100, binrange=(x_min, x_max), legend=True, kde_kws={'clip': (x_min, x_max)}, ax=axes[0]
).set(title='Protein size Distribution')

plot2 = sns.histplot(
    data=ligand, kde=True, bins=100, binrange=(0, 100), legend=True, kde_kws={'clip': (0, 100)}, ax=axes[1]
).set(title='Ligand size Distribution')

# plot.set_axis_labels("Size", "Count")
# ax = plot.axes[0][0]
# plt.xlim([x_min, x_max])
# plt.savefig('Plots/lico_target_dist.png', bbox_inches='tight')
plt.show()
