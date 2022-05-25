from pysmiles import read_smiles
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pdb

file = 'data/pdbind.csv'
target2 = []
with open(file, 'r') as file:
    filecontent = csv.reader(file)
    next(filecontent)
    for row in filecontent:
        prop = row[3]
        target2.append(float(prop))

x_min = 0
x_max = 15

z = {}
z['Target'] = np.array(target2)
plot = sns.displot(
    data=z, kde=True, kind="hist", bins=100, binrange=(x_min, x_max), legend=True, aspect=1.5, kde_kws={'clip': (x_min, x_max)}
).set(title='Target Property Distribution Distribution')

plot.set_axis_labels("-log(Kd/Ki)", "Count")
ax = plot.axes[0][0]
plt.xlim([x_min, x_max])
plt.savefig('Plots/pdbbind.png', bbox_inches='tight')

plt.show()