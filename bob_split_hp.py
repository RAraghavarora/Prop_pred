# NN model
import sys
import os
import pdb
from os import path, mkdir, chdir
import warnings
import numpy as np
import matplotlib.pyplot as plt
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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
        data_dir = '/scratch/ws/1/medranos-DFTBprojects/raghav/data/'
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
    bob_repr = np.array(
        [
            generate_bob(
                Z[mol],
                xyz[mol],
                atomtypes={'C', 'H', 'N', 'O', 'S', 'Cl'},
                asize={'C': 7, 'H': 16, 'N': 3, 'O': 3, 'S': 1, 'Cl': 2},
            )
            for mol in idx2
        ]
    )

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
                    bob_repr[ii],
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


def split_data(n_train, n_val, n_test, Repre, Target):
    # Training
    print("Perfoming training")

    X_train, X_val, X_test = (
        np.array(Repre[:n_train]),
        np.array(Repre[-n_test - n_val: -n_test]),
        np.array(Repre[-n_test:]),
    )
    Y_train, Y_val, Y_test = (
        np.array(Target[:n_train]),
        np.array(Target[-n_test - n_val: -n_test]),
        np.array(Target[-n_test:]),
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
    def __init__(self, params):
        super(NeuralNetwork, self).__init__()

        self.lin1 = nn.Linear(528, params['l1'])
        self.lin2 = nn.Linear(params['l1'] + 40, params['l2'])
        # self.lin3 = nn.Linear(128, 32)
        self.lin4 = nn.Linear(32, 1)
        self.apply(init_weights)
        # self.flatten = nn.Flatten(-1,0)

    def forward(self, x):
        slatm = x[:, :528]
        elec = x[:, 528:]
        layer1 = self.lin1(slatm)
        # layer1 = nn.functional.elu(layer1)

        concat = torch.cat([layer1, elec], dim=1)
        concat = nn.functional.elu(concat)

        layer2 = self.lin2(concat)
        layer2 = nn.functional.elu(layer2)
        # layer3 = self.lin3(layer2)
        # layer3 = nn.functional.elu(layer3)
        layer4 = self.lin4(layer3)

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


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience, parmas, model):
    batch_size = 16
    trainX, trainY, valX, valY, testX, testY = split_data(
        n_train, n_val, n_test, iX, iY
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
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.59, patience=500, min_lr=1e-6)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    epochs = 5000
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

    return test_mae


def objective(trial):

    params = {'l1': trial.suggest_categorical("l1", [16, 32, 64, 128]),
              'l2': trial.suggest_categorical("l2", [2, 4, 8, 16, 32, 64]),
              }

    model = NeuralNetwork(params)
    n_train = 10000
    n_val = 5000
    n_test = 41537 - 10000 - 5000
    patience = 700
    op = 'EAT'
    iX, iY = prepare_data(op)

    test_mae = fit_model_dense(
        n_train, n_val, n_test, iX, iY, patience, params, model)

    return test_mae[1]


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.RandomSampler(),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=30, interval_steps=10, n_min_trials=1000)
)
study.optimize(objective, n_trials=30)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))
