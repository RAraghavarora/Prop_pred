import pdb
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

file = 'data/lip_filter.csv'
pro_size = []
lig_size = []
targets = []
sizes = []
with open(file, 'r') as file:
    next(file)
    filecontent = csv.reader(file)
    for row in filecontent:
        # pro_size.append(int(row[1]))
        # lig_size.append(int(row[2]))
        if int(row[3])>90:
            continue
        targets.append(float(row[1]))
        sizes.append(int(row[3]))

# plt.subplot(1,2,1)
plt.plot(sizes, targets, 'x')
plt.xlabel('Size')
plt.ylabel('logD')
plt.title('Lipophilicity')

# plt.subplot(1,2,2)
# plt.plot(lig_size, targets, '.')
# plt.xlabel('Ligand Size')
# plt.ylabel('-log(Kd/Ki)')

plt.show()