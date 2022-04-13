import math
import sys
import os
import pdb
from os import path, mkdir, chdir
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


from qml.representations import (
    get_slatm_mbtypes,
    generate_slatm,
    generate_bob
)
import logging
import schnetpack as spk
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random


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
    logging.info("get dataset")

    # data preparation
    try:
        data_dir = '/scratch/ws/1/medranos-TUDprojects/raghav/data/'
        # data_dir = '../'
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
        ATOMS.append(atoms)
        AE.append(float(props['EAT']))
        EGAP.append(float(props['EGAP']))
        KSE.append(props['KSE'])
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

    AE = np.array(AE)
    EGAP = np.array(EGAP)
    TPROP = np.array(TPROP)

    # Generate representations
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

    reps2 = []
    for ii in range(len(idx2)):
        reps2.append(
            np.concatenate(
                (
                    slatm[ii],
                    p1b[ii],
                    p2b[ii],
                    p3b[ii],
                    p4b[ii],
                    p5b[ii],
                    p6b[ii],
                    p7b[ii],
                    p8b[ii],
                    np.linalg.norm(p9b[ii]),
                    p10b[ii],
                    p11b[ii],
                ),
                axis=None,
            )
        )

    return np.array(reps2), np.array(TPROP2)


def split_data(n_train, n_val, n_test, Repre, Target, seed):
    # Training
    print("Perfoming training")
    Repre = np.array(Repre)
    Target = np.array(Target)

    idx = np.arange(len(Target))
    np.random.seed(seed)
    idx2 = np.random.permutation(idx)

    X_train, X_val, X_test = (
        Repre[idx2[:n_train]],
        Repre[idx2[n_train: n_train + n_val]],
        Repre[idx2[n_train + n_val:]],
    )
    Y_train, Y_val, Y_test = (
        Target[idx2[:n_train]],
        Target[idx2[n_train: n_train + n_val]],
        Target[idx2[n_train + n_val:]],
    )

    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.lin1 = nn.Linear(17895, 16)
        self.lin2 = nn.Linear(56, 2)
        self.lin4 = nn.Linear(2, 1)
        self.apply(init_weights)
        # self.flatten = nn.Flatten(-1,0)

    def forward(self, x):
        slatm = x[:, :17895]
        elec = x[:, 17895:]
        layer1 = self.lin1(slatm)
        layer1 = nn.functional.leaky_relu(layer1)

        concat = torch.cat([layer1, elec], dim=1)
        # concat = nn.functional.elu(concat)

        layer2 = self.lin2(concat)
        layer2 = nn.functional.leaky_relu(layer2)
        layer4 = self.lin4(layer2)

        return layer4


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
    # device = "cpu"
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


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=7000, min_delta=0.001):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def plotting_results(model, test_loader, fold):
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

    out2 = open(f'errors_test_{fold}.dat', 'w')
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
    ctest = open(f'comp-test_{fold}.dat', 'w')
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
    plt.ylabel("True EGAP")
    plt.xlabel("Predicted EGAP")
    plt.savefig(f'Result{fold}.png')
    plt.close()


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience, split):
    batch_size = 32
    results = {}

    epochs = 20000
    # Randomly generated seeds np.random.seed(42); np.random.randint(50000, size=(22))
    seeds = [15795, 860, 38158, 44732, 11284, 6265, 16850, 37194, 21962,
             47191, 44131, 16023, 41090, 1685, 769, 2433, 5311, 37819,
             39188, 17568, 19769, 28693]

    for fold in range(0, split):
        print(f'FOLD {fold}')
        seed = seeds[fold]

        trainX, trainY, valX, valY, testX, testY = split_data(
            n_train, n_val, n_test, iX, iY, seed
        )

        X_train, X_val, X_test = (
            torch.from_numpy(trainX).float(),
            torch.from_numpy(valX).float(),
            torch.from_numpy(testX).float(),
        )

        Y_train, Y_val, Y_test = (
            torch.from_numpy(trainY).float(),
            torch.from_numpy(valY).float(),
            torch.from_numpy(testY).float(),
        )

        train = torch.utils.data.TensorDataset(X_train, Y_train)
        test = torch.utils.data.TensorDataset(X_test, Y_test)
        valid = torch.utils.data.TensorDataset(X_val, Y_val)
        # data loader
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        model = NeuralNetwork().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(
            optimizer, factor=0.5, patience=300, min_lr=1e-6)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        early_stopping = EarlyStopping()
        val_losses, val_errors, lrates = [], [], []
        for t in range(epochs):
            # print(f"Epoch {t+1}\n-------------------------------")
            train_nn(train_loader, model, loss_fn, optimizer)
            valid_loss, valid_mae = test_nn(valid_loader, model, loss_fn)
            # print(f"Validation MAE: {valid_mae}\n")
            scheduler.step(valid_mae)
            val_losses.append(valid_loss)
            val_errors.append(valid_mae)
            lrates.append(optimizer.param_groups[0]['lr'])
            early_stopping(valid_loss)
            if early_stopping.early_stop:
                break

        loss, test_mae = test_nn(test_loader, model, loss_fn)
        try:
            results[fold] = test_mae.item()
        except:
            results[fold] = test_mae
        torch.save(model.state_dict(), f'model_dict_{fold}.pt')

        lhis = open(f'learning-history_{fold}.dat', 'w')
        for ii in range(0, len(lrates)):
            lhis.write(
                '{:8d}'.format(ii)
                + '{:16f}'.format(lrates[ii])
                + '{:16f}'.format(val_losses[ii])
                + '{:16f}'.format(val_errors[ii])
                + '\n'
            )
        lhis.close()

        plotting_results(model, test_loader, fold)

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {split} FOLDS')
    print('--------------------------------')
    print(results)
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')

    return (
        model,
        results
    )


train_set = [25000, 30000]
splits = [1]
op = 'EAT'
n_val = 6000

iX, iY = prepare_data(op)

patience = 500

current_dir = os.getcwd()

for ii in range(len(train_set)):
    n_test = len(iY) - n_val
    print('Trainset= {:}'.format(train_set[ii]))
    chdir(current_dir)
    os.chdir(current_dir + '/withdft/kcv/PureRandom')
    try:
        os.mkdir(str(train_set[ii]))
    except:
        pass
    os.chdir(current_dir + '/withdft/kcv/PureRandom/' + str(train_set[ii]))

    split = splits[ii]
    n_train = int(train_set[ii])

    model, results = fit_model_dense(
        n_train, n_val, n_test, iX, iY, patience, split
    )
    outfile = open('output.txt', 'w')
    import json
    outfile.write(json.dumps(results))
    outfile.close()
    # lhis = open('learning-history.dat', 'w')
    # for ii in range(0, len(lr)):
    #     lhis.write(
    #         '{:8d}'.format(ii)
    #         + '{:16f}'.format(lr[ii])
    #         + '{:16f}'.format(loss[ii])
    #         + '{:16f}'.format(mae[ii])
    #         + '\n'
    #     )
    # lhis.close()

    # Saving NN model
    # torch.save(model, 'model.pt')

    # Saving results
    # plotting_results(model, test_loader)
