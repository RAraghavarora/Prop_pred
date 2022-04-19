import optuna
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
# from qml.representations import (
#     get_slatm_mbtypes,
#     generate_slatm,
# )
import warnings
from sklearn.preprocessing import StandardScaler


def complete_array(Aprop, maxsize=90):
    Aprop2 = []
    for ii in range(len(Aprop)):
        n1 = len(Aprop[ii])
        if n1 == maxsize:
            Aprop2.append(Aprop[ii])
        else:
            n2 = maxsize - n1
            Aprop2.append(np.concatenate((Aprop[ii], np.zeros(n2)), axis=None))

    return Aprop2


def prepare_data():
    try:
        npzdir = '/scratch/ws/1/medranos-TUDprojects/raghav/data/lipo/npz/'
        files = [f for f in os.listdir(npzdir) if os.path.isfile(npzdir + f)]
    except:
        npzdir = './data/lipo/npz/'
        files = [f for f in os.listdir(npzdir) if os.path.isfile(npzdir + f)]
    target, xyz, Z = [], [], []
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = ([] for i in range(11))

    n = len(files)

    for fileindex in range(n):
        molecule = np.load(npzdir + files[fileindex])
        size = molecule['size']
        if size > 90:
            continue
        Z.append(molecule['Z'])
        xyz.append(molecule['XYZ'])
        target.append(float(molecule['target']))
        p1.append(float(molecule['EFermi']))
        p2.append(float(molecule['EBand']))
        p3.append(float(molecule['NE']))
        p4.append(float(molecule['Eh0']))
        p5.append(float(molecule['Escc']))
        p6.append(float(molecule['E3rd']))
        p7.append(float(molecule['Erep3']))
        p8.append(float(molecule['Embd']))
        p9.append(molecule['dip'])
        p10.append(molecule['Eig'])
        p11.append(molecule['CHG'])

    target = np.array(target)
    n = len(target)
    print(n)
    idx = np.arange(n)
    np.random.seed(2314)
    idx2 = np.random.permutation(idx)

    p1b, p2b, p3b, p4b, p5b, p6b, p7b, p8b, p9b, p10b, p11b = (
        [] for i in range(11))
    target2 = []

    for nn1 in idx2[:n]:
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
        target2.append(target[nn1])

    p11b = complete_array(p11)

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

    # Not standardizing the charges, because a lot of them are 0
    p1b, p2b, p3b, p4b, p5b, p6b, p7b, p8b, p9b, p10b, _ = temp

    # mbtypes = get_slatm_mbtypes([Z[mol] for mol in idx2[:n]])
    # slatm = [
    #     generate_slatm(mbtypes=mbtypes,
    #                    nuclear_charges=Z[mol], coordinates=xyz[mol])
    #     for mol in idx2[:n]
    # ]
    # slatm_len = len(slatm[0])

    reps2 = []
    for ii in range(len(idx2[:n])):
        reps2.append(
            np.concatenate(
                (
                    # slatm[ii],
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

    return np.array(reps2), np.array(target2)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class NeuralNetwork(nn.Module):
    def __init__(self, params):
        super(NeuralNetwork, self).__init__()

        self.lin1 = nn.Linear(107, params['l1'])
        self.lin2 = nn.Linear(params['l1'], params['l2'])
        self.lin4 = nn.Linear(params['l2'], 1)
        self.apply(init_weights)
        # self.flatten = nn.Flatten(-1,0)

    def forward(self, x):

        layer1 = self.lin1(x)
        layer1 = nn.functional.leaky_relu(layer1)

        layer2 = self.lin2(layer1)
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

    def __init__(self, patience=3000, min_delta=0.005):
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
        if self.best_loss is None:
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


def split_data(n_train, n_val, n_test, Repre, Target):
    # Training
    print("Perfoming training")

    X_train, X_val, X_test = (
        np.array(Repre[:n_train]),
        np.array(Repre[n_train: n_train + n_val]),
        np.array(Repre[n_train + n_val:]),
    )
    Y_train, Y_val, Y_test = (
        np.array(Target[:n_train]),
        np.array(Target[n_train: n_train + n_val]),
        np.array(Target[n_train + n_val:]),
    )

    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience, model, trial):
    batch_size = 32
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
        optimizer, factor=0.50, patience=100, min_lr=1e-6)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    early_stopping = EarlyStopping()

    epochs = 1
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
        early_stopping(valid_mae)
        if early_stopping.early_stop:
            warnings.warn(f"Stopping early after {t+1} epochs for {n_train}")
            break
        trial.report(valid_mae, t)

        if trial.should_prune():
            print("Pruning")
            raise optuna.TrialPruned()

    test_mae = test_nn(test_loader, model, loss_fn)
    print(
        f"Finished training on train_size={n_train}\n Testing MAE = {test_mae}")

    return test_mae


def objective(trial):

    iX, iY = prepare_data()

    params = {'l1': trial.suggest_categorical("l1", [4, 8, 16, 32, 64, 128]),
              'l2': trial.suggest_categorical("l2", [4, 8, 16, 32, 64]),
              }

    model = NeuralNetwork(params)
    n_train = 2000
    n_val = 500
    n_test = 1592
    patience = 100

    test_mae = fit_model_dense(
        n_train, n_val, n_test, iX, iY, patience, model, trial)

    return test_mae[1]


study = optuna.create_study(
    study_name="lipo",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=2, n_warmup_steps=30, interval_steps=10)
)
study.optimize(objective, n_trials=30, n_jobs=-1, timeout=86400)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))