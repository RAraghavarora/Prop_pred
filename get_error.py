import pdb
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
import logging
import schnetpack as spk
from sklearn.preprocessing import StandardScaler


def complete_array(Aprop):
    Aprop2 = []
    for ii in range(len(Aprop)):
        n1 = len(Aprop[ii])
        if n1 == 23:
            Aprop2.append(Aprop[ii])
        else:
            n2 = 23 - n1
            Aprop2.append(np.concatenate((Aprop[ii], np.zeros(n2)), axis=None))

    return Aprop2


def prepare_data(op):
    #  # read dataset
    properties = [
        'RMSD',
        'EAT',
        'EMBD',
        'EGAP',
        'KSE',
        'FermiEne',
        'BandEne',
        'NumElec',
        'h0Ene',
        'sccEne',
        '3rdEne',
        'RepEne',
        'mbdEne',
        'TBdip',
        'TBeig',
        'TBchg',
    ]

    # data preparation
    logging.info("get dataset")
    try:
        data_dir = '/scratch/ws/1/medranos-DFTBprojects/raghav/data/'
        dataset = spk.data.AtomsData(
            data_dir + 'distort.db', load_only=properties)
    except:
        data_dir = '../'
        dataset = spk.data.AtomsData(
            data_dir + 'totgdb7x_pbe0.db', load_only=properties)

    n = len(dataset)
    print(n)
    idx = np.arange(n)
    np.random.seed(2314)
    idx2 = np.random.permutation(idx)

    # computing predicted property
    logging.info("get predicted property")
    AE, xyz, Z = [], [], []
    EGAP, KSE, TPROP = [], [], []
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    ATOMS = []
    for i in idx2[:n]:
        atoms, props = dataset.get_properties(int(i))
        AE.append(float(props['EAT']))
        EGAP.append(float(props['EGAP']))
        KSE.append(props['KSE'])
        TPROP.append(float(props[op]))
        TPROP.append(float(props[op]))
        xyz.append(atoms.get_positions())
        Z.append(atoms.get_atomic_numbers())
        p1.append(float(props['FermiEne']))
        p2.append(float(props['BandEne']))
        p3.append(float(props['NumElec']))
        p4.append(float(props['h0Ene']))
        p5.append(float(props['sccEne']))
        p6.append(float(props['3rdEne']))
        p7.append(float(props['RepEne']))
        p8.append(float(props['mbdEne']))
        p9.append(props['TBdip'])
        p10.append(props['TBeig'])
        p11.append(props['TBchg'])

    print('extracted the properties')

    AE = np.array(AE)
    EGAP = np.array(EGAP)
    TPROP = np.array(TPROP)

    TPROP2 = []
    p1b, p2b, p11b, p3b, p4b, p5b, p6b, p7b, p8b, p9b, p10b = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for nn1 in idx2:
        p1b.append(p1[nn1])
        p2b.append(p2[nn1])
        p3b.append(p3[nn1])
        p4b.append(p4[nn1])
        p5b.append(p5[nn1])
        p6b.append(p6[nn1])
        p7b.append(p7[nn1])
        p8b.append(p8[nn1])
        p9b.append(p9[nn1].numpy())
        p10b.append(p10[nn1].numpy())
        p11b.append(p11[nn1].numpy())
        TPROP2.append(TPROP[nn1])

    p11b = complete_array(p11b)

    # Standardize the data property wise

    temp = []
    for var in [p1b, p2b, p3b, p4b, p5b, p6b, p7b, p8b, p9b, p10b, p11b]:
        var2 = np.array(var)
        try:
            _ = var2.shape[1]
        except IndexError:
            var2 = var2.reshape(-1, 1)
        scaler = StandardScaler()
        var3 = scaler.fit_transform(var2)

        temp.append(var3)

    p1b, p2b, p3b, p4b, p5b, p6b, p7b, p8b, p9b, p10b, p11b = temp

    global_features = []
    for ii in range(len(idx2)):
        global_features.append(
            np.concatenate(
                (
                    p1b[ii],
                    p2b[ii],
                    p3b[ii],
                    p4b[ii],
                    p5b[ii],
                    p6b[ii],
                    p7b[ii],
                    p8b[ii],
                ),
                axis=None,
            )
        )
    global_features = np.array(global_features)

    return [global_features, p9b, p10b, p11b], TPROP2


class NeuralNetwork(nn.Module):
    def __init__(self, l1=16, l2=2, l3=16, l4=2, l5=16):
        super(NeuralNetwork, self).__init__()

        self.lin1 = nn.Linear(8, l1)
        self.lin2 = nn.Linear(3, l2)
        self.lin3 = nn.Linear(8, l3)
        self.lin4 = nn.Linear(23, l4)

        self.lin5 = nn.Linear(l1 + l2 + l3 + l4, l5)
        self.lin6 = nn.Linear(l5, 1)

        self.apply(init_weights)
        # self.flatten = nn.Flatten(-1,0)

    def forward(self, x):
        # x = self.flatten(x)
        global_features, p9b, p10b, p11b = x[:,0:8], x[:, 8:11], x[:, 11:19], x[:, 19:]
        layer1 = self.lin1(global_features)
        layer1 = nn.functional.elu(layer1)
        layer2 = self.lin2(p9b)
        layer2 = nn.functional.elu(layer2)
        layer3 = self.lin3(p10b)
        layer3 = nn.functional.elu(layer3)
        layer4 = self.lin4(p11b)
        layer4 = nn.functional.elu(layer4)

        concat = torch.cat([layer1, layer2, layer3, layer4], dim=1)
        concat = nn.functional.elu(concat)

        layer5 = self.lin5(concat)
        layer5 = nn.functional.elu(layer5)
        layer6 = self.lin6(layer5)

        return layer6


def split_data(n_train, n_val, n_test, Repre, Target):
    # Training
    print("Perfoming training")
    global_features, p9b, p10b, p11b = Repre

    X_train1, X_val1, X_test1 = (
        torch.from_numpy(np.array(global_features[:n_train])).float(),
        torch.from_numpy(
            np.array(global_features[-n_test - n_val: -n_test])).float(),
        torch.from_numpy(np.array(global_features[-n_test:])).float(),
    )

    X_train2, X_val2, X_test2 = (
        torch.from_numpy(np.array(p9b[:n_train])).float(),
        torch.from_numpy(np.array(p9b[-n_test - n_val: -n_test])).float(),
        torch.from_numpy(np.array(p9b[-n_test:])).float(),
    )

    X_train3, X_val3, X_test3 = (
        torch.from_numpy(np.array(p10b[:n_train])).float(),
        torch.from_numpy(np.array(p10b[-n_test - n_val: -n_test])).float(),
        torch.from_numpy(np.array(p10b[-n_test:])).float(),
    )

    X_train4, X_val4, X_test4 = (
        torch.from_numpy(np.array(p11b[:n_train])).float(),
        torch.from_numpy(np.array(p11b[-n_test - n_val: -n_test])).float(),
        torch.from_numpy(np.array(p11b[-n_test:])).float(),
    )

    Y_train, Y_val, Y_test = (
        torch.from_numpy(np.array(Target[:n_train])).float(),
        torch.from_numpy(np.array(Target[-n_test - n_val: -n_test])).float(),
        torch.from_numpy(np.array(Target[-n_test:])).float(),
    )

    # Data standardization
    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return [X_train1, X_train2, X_train3, X_train4], Y_train, [X_val1, X_val2, X_val3, X_val4], Y_val, [X_test1, X_test2, X_test3, X_test4], Y_test


op = 'EGAP'
iX, iY = prepare_data(op)
n_train = 30000
n_val = 5000
n_test = len(iY) - n_train - n_val
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(
    n_train, n_val, n_test, iX, iY
)
test = torch.utils.data.TensorDataset(torch.cat(X_test, dim=1), Y_test)
# data loader
test_loader = DataLoader(test, batch_size=16, shuffle=False)
model = torch.load('only_dft/egap/30000/model.pt', map_location='cuda:0')

with torch.no_grad():
    x = test_loader.dataset.tensors[0].cuda()
    model.eval()
    pred = model(x)
    y = test_loader.dataset.tensors[1].cuda()
    loss_fn = nn.MSELoss()
    test_loss = loss_fn(pred, y).item()
    mae_loss = torch.nn.L1Loss(reduction='mean')
    mae = mae_loss(pred, y)
    print(mae)

    dtest = np.array(pred.cpu() - y.cpu())
    Y_test = y.reshape(-1, 1)
    format_list1 = ['{:16f}' for item1 in Y_test[0]]
    s = ' '.join(format_list1)
    ctest = open('only_dft/egap/30000/comp.dat', 'w')
    for ii in range(0, len(pred)):
        ctest.write(
            s.format(*pred[ii]) + s.format(*Y_test[ii]) +
            s.format(*dtest[ii]) + '\n'
        )
    ctest.close()
