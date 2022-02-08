import sys
import os
import pdb
from os import path, mkdir, chdir
import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import logging
import schnetpack as spk
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.metrics import MeanAbsoluteError
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping


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

    # data preparation
    logging.info("get dataset")
    try:
        data_dir = '/scratch/ws/1/medranos-DFTBprojects/raghav/data/'
        dataset = spk.data.AtomsData(data_dir + 'distort.db', load_only=properties)
    except:
        data_dir = '../'
        dataset = spk.data.AtomsData(data_dir + 'totgdb7x_pbe0.db', load_only=properties)
        
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
        p9b.append(p9[nn1])
        p10b.append(p10[nn1])
        p11b.append(p11[nn1])
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


def split_data(n_train, n_val, n_test, Repre, Target):
    # Training
    print("Perfoming training")
    global_features, p9b, p10b, p11b = Repre

    X_train1, X_val1, X_test1 = (
        torch.from_numpy(np.array(global_features[:n_train])).float(),
        torch.from_numpy(np.array(global_features[-n_test - n_val: -n_test])).float(),
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

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

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
        global_features, p9b, p10b, p11b = x[:,0:8], x[:,8:11], x[:,11:19], x[:,19:]
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


def score_function(engine):
    val_loss = engine.state.metrics['mae']
    return val_loss


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience):
    batch_size = 16
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(
        n_train, n_val, n_test, iX, iY
    )
    train = torch.utils.data.TensorDataset(torch.cat(X_train, dim=1),Y_train)
    test = torch.utils.data.TensorDataset(torch.cat(X_test, dim=1),Y_test)
    valid = torch.utils.data.TensorDataset(torch.cat(X_val, dim=1),Y_val)
    # data loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle = False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle = False)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle = False)

    # device = "cuda"
    device = "cpu"
    model = NeuralNetwork().to(device)
    model = nn.DataParallel(model)
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.57, patience = 500, min_lr=1e-6)

    epochs = 20000

    trainer = create_supervised_trainer(model, optimizer, loss_fn)
    val_metrics = {
        "mae": MeanAbsoluteError(),
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics)
    MeanAbsoluteError().attach(evaluator, "mae")
    handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)


    validate_every = 10

    @trainer.on(Events.EPOCH_COMPLETED(every=validate_every))
    def run_validation():
        evaluator.run(valid_loader)

    @trainer.on(Events.EPOCH_COMPLETED(every=validate_every))
    def log_validation():
        metrics = evaluator.state.metrics
        print(f"Epoch: {trainer.state.epoch},  MAE: {metrics['mae']}")

    trainer.run(train_loader, epochs)

    x = evaluator.run(test_loader)
    print(x)
    pdb.set_trace()

    return model

# prepare dataset
train_set = ['59']
op = 'EGAP'
n_val = 5000

iX, iY = prepare_data(op)

# fit model and plot learning curves for a patience
patience = 500

current_dir = os.getcwd()

for ii in range(len(train_set)):
    n_test = len(iY) - int(train_set[ii]) - n_val
    print('Trainset= {:}'.format(train_set[ii]))
    chdir(current_dir)
    os.chdir(current_dir + '/only_dft/egap/')
    try:
        os.mkdir(str(train_set[ii]))
    except:
        pass
    os.chdir(current_dir + '/only_dft/egap/' + str(train_set[ii]))

    model = fit_model_dense(
        int(train_set[ii]), int(n_val), int(n_test), iX, iY, patience
    )

    # Saving NN model
    torch.save(model, 'model.pt')
