import sys
import os
import pdb
from os import path, mkdir, chdir
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from qml.representations import (
    get_slatm_mbtypes,
    generate_slatm,
)

import logging
import schnetpack as spk
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# monitor the learning rate


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


# prepare train and test dataset


def prepare_data(op):
    #  # read dataset
    properties = [
        'EAT',
        'EGAP',
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
        data_dir = '/scratch/ws/1/medranos-TUDprojects/raghav/data/'
        dataset = spk.data.AtomsData(
            data_dir + 'qm7x-eq-n1.db', load_only=properties)
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
        ATOMS.append(atoms)
        AE.append(float(props['EAT']))
        EGAP.append(float(props['EGAP']))
        TPROP.append(float(props[op] * 23.0621))
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

    AE = np.array(AE)
    EGAP = np.array(EGAP)
    TPROP = np.array(TPROP)

    mbtypes = get_slatm_mbtypes([Z[mol] for mol in idx2])
    slatm = [
        generate_slatm(mbtypes=mbtypes,
                       nuclear_charges=Z[mol], coordinates=xyz[mol])
        for mol in idx2
    ]

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
    desc = []
    for ii in range(len(idx2)):
        desc.append(slatm[ii])
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
                    np.linalg.norm(p9b[ii]),
                ),
                axis=None,
            )
        )
    global_features = np.array(global_features)

    return [desc, global_features, p10b, p11b], TPROP2


def split_data(n_train, n_val, n_test, Repre, Target):
    # Training
    print("Perfoming training")
    desc, global_features, p10b, p11b = Repre

    X_train0, X_val0, X_test0 = (
        torch.from_numpy(np.array(desc[:n_train])).float(),
        torch.from_numpy(np.array(desc[n_train: n_train + n_val])).float(),
        torch.from_numpy(np.array(desc[n_train + n_val:])).float(),
    )

    X_train1, X_val1, X_test1 = (
        torch.from_numpy(np.array(global_features[:n_train])).float(),
        torch.from_numpy(
            np.array(global_features[n_train: n_train + n_val])).float(),
        torch.from_numpy(np.array(global_features[n_train + n_val:])).float(),
    )

    X_train3, X_val3, X_test3 = (
        torch.from_numpy(np.array(p10b[:n_train])).float(),
        torch.from_numpy(np.array(p10b[n_train: n_train + n_val])).float(),
        torch.from_numpy(np.array(p10b[n_train + n_val:])).float(),
    )

    X_train4, X_val4, X_test4 = (
        torch.from_numpy(np.array(p11b[:n_train])).float(),
        torch.from_numpy(np.array(p11b[n_train: n_train + n_val])).float(),
        torch.from_numpy(np.array(p11b[n_train + n_val:])).float(),
    )

    Y_train, Y_val, Y_test = (
        torch.from_numpy(np.array(Target[:n_train])).float(),
        torch.from_numpy(np.array(Target[n_train: n_train + n_val])).float(),
        torch.from_numpy(np.array(Target[n_train + n_val:])).float(),
    )

    # Data standardization
    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return [X_train0, X_train1, X_train3, X_train4], Y_train, [X_val0, X_val1, X_val3, X_val4], Y_val, [X_test0, X_test1, X_test3, X_test4], Y_test


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.lin0 = nn.Linear(17932, 16)
        self.lin1 = nn.Linear(16, 4)
        self.lin2 = nn.Linear(4, 1)

        self.apply(init_weights)
        # self.flatten = nn.Flatten(-1,0)

    def forward(self, x):
        # x = self.flatten(x)
        warnings.warn(str(x.device))
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        dlen = 17895
        desc, global_features, p10b, p11b = (
            x[:, 0:dlen],
            x[:, dlen:dlen + 9],
            x[:, dlen + 9:dlen + 17],
            x[:, dlen + 17:]
        )

        # Convolve the matrices
        input_vector = []
        for a_desc, a_glo, a_p10, a_p11 in zip(desc, global_features, p10b, p11b):
            temp1 = np.convolve(a_p10.cpu(), a_p11.cpu())
            temp2 = np.convolve(a_glo.cpu(), temp1)
            temp3 = np.convolve(a_desc.cpu(), temp2)
            input_vector.append(temp3)

        input_vector = torch.from_numpy(np.array(input_vector)).to(device)

        layer0 = self.lin0(input_vector)
        layer0 = nn.functional.elu(layer0)
        layer1 = self.lin1(layer0)
        layer1 = nn.functional.elu(layer1)
        layer2 = self.lin2(layer1)

        return layer2


def train_nn(dataloader, model, loss_fn, optimizer):
    # size = len(dataloader.dataset)
    model.train()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # mae = float(mean_absolute_error(pred,y))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_nn(dataloader, model, loss_fn):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, mae = 0, 0
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            mae_loss = torch.nn.L1Loss(reduction='mean')
            mae += mae_loss(pred, y)

    test_loss /= num_batches
    mae /= num_batches
    return test_loss, mae


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience):
    batch_size = 16
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(
        n_train, n_val, n_test, iX, iY
    )
    train = torch.utils.data.TensorDataset(torch.cat(X_train, dim=1), Y_train)
    test = torch.utils.data.TensorDataset(torch.cat(X_test, dim=1), Y_test)
    valid = torch.utils.data.TensorDataset(torch.cat(X_val, dim=1), Y_val)
    # data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model = NeuralNetwork().to(device)
    model = nn.DataParallel(model)
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.5, patience=50, min_lr=1e-6)

    epochs = 20000
    val_losses, val_errors, lrates = [], [], []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_nn(train_loader, model, loss_fn, optimizer)
        valid_loss, valid_mae = test_nn(valid_loader, model, loss_fn)
        print(f"Validation MAE: {valid_mae}\n")
        scheduler.step(valid_mae)
        val_losses.append(valid_loss)
        val_errors.append(valid_mae)
        lrates.append(optimizer.param_groups[0]['lr'])

    test_mae = test_nn(test_loader, model, loss_fn)
    print(
        f"Finished training on train_size={n_train}\n Testing MAE = {test_mae}")

    return (
        model,
        lrates,
        val_losses,
        val_errors,
        test_loader
    )


def plotting_results(model, test_loader):
    # applying nn model
    with torch.no_grad():
        x = test_loader.dataset.tensors[0].cuda()
        pred = model(x)
        y = test_loader.dataset.tensors[1].cuda()
        loss_fn = nn.MSELoss()
        test_loss = loss_fn(pred, y).item()
        mae_loss = torch.nn.L1Loss(reduction='mean')
        mae = mae_loss(pred, y)

    STD_PROP = float(pred.std())

    out2 = open('errors_test.dat', 'w')
    out2.write(
        '{:>24}'.format(STD_PROP)
        + '{:>24}'.format(mae)
        + '{:>24}'.format(test_loss)
        + "\n"
    )
    out2.close()

    # writing ouput for comparing values
    dtest = np.array(pred.cpu() - y.cpu())
    Y_test = y.reshape(-1, 1)
    format_list1 = ['{:16f}' for item1 in Y_test[0]]
    s = ' '.join(format_list1)
    ctest = open('comp-test.dat', 'w')
    for ii in range(0, len(pred)):
        ctest.write(
            s.format(*pred[ii]) + s.format(*Y_test[ii]) +
            s.format(*dtest[ii]) + '\n'
        )
    ctest.close()

    # Save as a plot
    plt.plot(pred.cpu(), y.cpu(), '.')
    mini = min(y).item()
    maxi = max(y).item()
    temp = np.arange(mini, maxi, 0.1)
    plt.plot(temp, temp)
    plt.xlabel("True EGAP")
    plt.ylabel("Predicted EGAP")
    plt.savefig('Result.png')


print("Device count: ", torch.cuda.device_count())

# prepare dataset
train_set = ['1000', '20000', '4000']
op = 'EAT'
n_val = 5000

iX, iY = prepare_data(op)

# fit model and plot learning curves for a patience
patience = 500

current_dir = os.getcwd()

for ii in range(len(train_set)):
    n_test = len(iY) - n_val
    print('Trainset= {:}'.format(train_set[ii]))
    chdir(current_dir)
    os.chdir(current_dir + '/withdft/bob/eq/conv/')
    try:
        os.mkdir(str(train_set[ii]))
    except:
        pass
    os.chdir(current_dir + '/withdft/bob/eq/conv/' + str(train_set[ii]))

    model, lr, loss, mae, test_loader = fit_model_dense(
        int(train_set[ii]), int(n_val), int(n_test), iX, iY, patience
    )

    lhis = open('learning-history.dat', 'w')
    for ii in range(0, len(lr)):
        lhis.write(
            '{:8d}'.format(ii)
            + '{:16f}'.format(lr[ii])
            + '{:16f}'.format(loss[ii])
            + '{:16f}'.format(mae[ii])
            + '\n'
        )
    lhis.close()

    # Saving NN model
    torch.save(model, 'model.pt')

    # Saving results
    plotting_results(model, test_loader)
