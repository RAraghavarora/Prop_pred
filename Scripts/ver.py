import pdb
import numpy as np
import schnetpack as spk
data_dir = './data/'
dataset = spk.data.AtomsData(data_dir + 'qm9-dftb.db')
n = len(dataset)
idx = np.arange(n)
np.random.seed(2314)
idx2 = np.random.permutation(idx)
np.random.seed(2314)
idx2 = np.random.permutation(idx2)


AE = []
for i in idx2[:n]:
    atoms, props = dataset.get_properties(int(i))
    AE.append(float(props['EAT']) * 23.0621)

pdb.set_trace()
