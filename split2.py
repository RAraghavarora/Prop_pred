#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import pdb
from os import path, mkdir, chdir
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


import logging
import schnetpack as spk
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# In[17]:


def complete_array(Aprop):
    Aprop2 = []
    for ii in range(len(Aprop)):
        n1 = len(Aprop[ii])
        if n1 == 29:
            Aprop2.append(Aprop[ii])
        else:
            n2 = 29 - n1
            Aprop2.append(np.concatenate((Aprop[ii], np.zeros(n2)), axis=None))

    return Aprop2


# In[15]:


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
        data_dir = '/scratch/ws/1/medranos-DFTBprojects/raghav/data/'
        dataset = spk.data.AtomsData(
            data_dir + 'qm9-dftb.db', load_only=properties)
    except:
        data_dir = 'C:/raghav/Prop_pred/data/'
        dataset = spk.data.AtomsData(
            data_dir + 'qm9-dftb.db', load_only=properties)

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
                    np.linalg.norm(p9b[ii])
                ),
                axis=None,
            )
        )
    global_features = np.array(global_features)

    return [global_features, p10b, p11b], TPROP2


# In[4]:


def split_data(n_train, n_val, n_test, Repre, Target):
    # Training
    print("Perfoming training")
    global_features, p10b, p11b = Repre

    X_train1, X_val1, X_test1 = (
        torch.from_numpy(np.array(global_features[:n_train])).float(),
        torch.from_numpy(
            np.array(global_features[-n_test - n_val: -n_test])).float(),
        torch.from_numpy(np.array(global_features[-n_test:])).float(),
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

    return [X_train1, X_train3, X_train4], Y_train, [X_val1, X_val3, X_val4], Y_val, [X_test1, X_test3, X_test4], Y_test


# In[21]:


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class NeuralNetwork(nn.Module):
    def __init__(self, params):
        super(NeuralNetwork, self).__init__()
        print(params)
        self.lin1 = nn.Linear(9, params['l1'])
        self.lin3 = nn.Linear(8, params['l3'])
        self.lin4 = nn.Linear(29, params['l4'])

        self.lin5 = nn.Linear(
            params['l1'] + params['l3'] + params['l4'], params['l5'])
        self.lin6 = nn.Linear(params['l5'], 1)

        self.apply(init_weights)
        # self.flatten = nn.Flatten(-1,0)

    def forward(self, x):
        # x = self.flatten(x)
        global_features, p10b, p11b = x[:, 0:9], x[:, 9:17], x[:, 17:]
        layer1 = self.lin1(global_features)
        layer1 = nn.functional.elu(layer1)
        layer3 = self.lin3(p10b)
        layer3 = nn.functional.elu(layer3)
        layer4 = self.lin4(p11b)
        layer4 = nn.functional.elu(layer4)

        concat = torch.cat([layer1, layer3, layer4], dim=1)
        concat = nn.functional.elu(concat)

        layer5 = self.lin5(concat)
        layer5 = nn.functional.elu(layer5)
        layer6 = self.lin6(layer5)

        return layer6


# In[23]:


def train_nn(dataloader, model, loss_fn, optimizer):
    # size = len(dataloader.dataset)
    model.train()
    num_batches = len(dataloader)

    # device = "cuda"
    device = "cpu"
    mae = 0
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
        mae_loss = torch.nn.L1Loss(reduction='mean')
        mae += mae_loss(pred, y)

    mae /= num_batches
    return mae


def test_nn(dataloader, model, loss_fn):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, mae = 0, 0
    # device = "cuda"
    device = "cpu"
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


def plotting_results(model, test_loader):
    # applying nn model
    with torch.no_grad():
        x = test_loader.dataset.tensors[0]
        pred = model(x)
        y = test_loader.dataset.tensors[1]
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


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience, params, model, trial):
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

    # device = "cuda"
    device = "cpu"
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.6, patience=500, min_lr=1e-6)

    epochs = 10000
    val_losses, val_errors, lrates = [], [], []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        mae = train_nn(train_loader, model, loss_fn, optimizer)
#         valid_loss, valid_mae = test_nn(valid_loader, model, loss_fn)
        print(f"Train MAE: {mae}\n")
        scheduler.step(mae)

        trial.report(mae, t)

        if trial.should_prune():
            print("Pruning")
            raise optuna.TrialPruned()

#         val_losses.append(valid_loss)
#         val_errors.append(valid_mae)
        # lrates.append(optimizer.param_groups[0]['lr'])

    test_mae = test_nn(test_loader, model, loss_fn)
    print(
        f"Finished training on train_size={n_train}\n Testing MAE = {test_mae}")

    return (
        model,
        test_mae,
        test_loader
    )


# In[19]:


# def objective(trial):

#     params = {'l1': trial.suggest_categorical("l1", [2,4,8]),
#               'l3': trial.suggest_categorical("l3", [2,4,8]),
#               'l4': trial.suggest_categorical("l4", [2,4,8]),
#               'l5': trial.suggest_categorical("l5", [2,4,8])
#               }

#     model = NeuralNetwork(params)
#     n_train=10000
#     n_val=5000
#     n_test=42000
#     patience = 700
#     op = 'EGAP'
#     iX, iY = prepare_data(op)

#     test_mae = fit_model_dense(n_train, n_val, n_test, iX, iY, patience, params, model, trial)

#     return test_mae[0]


# # In[ ]:


# import optuna

# study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.MedianPruner(
#         n_startup_trials=2, n_warmup_steps=30, interval_steps=10))
# study.optimize(objective, n_trials=15, timeout=21600)


# # In[ ]:


# best_trial = study.best_trial

# for key, value in best_trial.params.items():
#     print("{}: {}".format(key, value))

# # learning_rate: 0.0018518678521842887
# # optimizer: Adam
# # n_unit: 9


# # In[ ]:


def objective():

    params = {'l1': 8,
              'l3': 16,
              'l4': 2,
              'l5': 16
              }

    model = NeuralNetwork(params)
    n_train = [30000, 1000, 20000, 10000, 8000, 4000, 2000]
    n_val = 5000
    n_test = 50000
    patience = 700
    op = 'EGAP'
    current_dir = os.getcwd()
    os.chdir(current_dir + 'only_dft/egap/qm9/')
    iX, iY = prepare_data(op)
    for train in n_train:
        os.mkdir(str(train))
        os.chdir(current_dir + 'only_dft/egap/qm9/' + str(train))
        model, test_mae, test_loader = fit_model_dense(
            train, n_val, n_test, iX, iY, patience, params, model)
        print(train, '\t', test_mae[0])

        torch.save(model, 'model.pt')
        plotting_results(model, test_loader)


objective()
