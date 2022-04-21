import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from qml.representations import (
    get_slatm_mbtypes,
    generate_slatm,
)
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
    npzdir = '/scratch/ws/1/medranos-TUDprojects/raghav/data/lipo/npz/'
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

    mbtypes = get_slatm_mbtypes([Z[mol] for mol in idx2[:n]])
    slatm = [
        generate_slatm(mbtypes=mbtypes,
                       nuclear_charges=Z[mol], coordinates=xyz[mol])
        for mol in idx2[:n]
    ]
    slatm_len = len(slatm[0])

    reps2 = []
    desc = []
    for ii in range(len(idx2[:n])):
        desc.append(slatm[ii])
        reps2.append(
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

    return [np.array(desc), np.array(reps2), np.array(p10b), np.array(p11b)], np.array(target2), slatm_len


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class NeuralNetwork(nn.Module):
    def __init__(self, slatm_len, l0=16, l1=16, l2=2, l3=16, l4=2, l5=16):
        super(NeuralNetwork, self).__init__()

        self.slatm_len = slatm_len
        self.lin0 = nn.Linear(slatm_len, l0)
        self.lin1 = nn.Linear(8, l1)
        self.lin3 = nn.Linear(8, l3)
        self.lin4 = nn.Linear(90, l4)

        self.lin5 = nn.Linear(l0 + l1 + l3 + l4, l5)
        self.lin6 = nn.Linear(l5, 1)

        self.apply(init_weights)
        # self.flatten = nn.Flatten(-1,0)

    def forward(self, x):
        dlen = self.slatm_len
        desc, global_features, p10b, p11b = (
            x[:, 0:dlen],
            x[:, dlen:dlen + 8],
            x[:, dlen + 8:dlen + 16],
            x[:, dlen + 16:]
        )

        layer0 = self.lin0(desc)
        layer0 = nn.functional.elu(layer0)
        layer1 = self.lin1(global_features)
        layer1 = nn.functional.elu(layer1)

        layer3 = self.lin3(p10b)
        layer3 = nn.functional.elu(layer3)
        layer4 = self.lin4(p11b)
        layer4 = nn.functional.elu(layer4)
        concat = torch.cat([layer0, layer1, layer3, layer4], dim=1)
        concat = nn.functional.elu(concat)
        layer5 = self.lin5(concat)
        layer5 = nn.functional.elu(layer5)
        layer6 = self.lin6(layer5)

        return layer6


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
    desc, global_features, p10b, p11b = Repre

    X_train0, X_val0, X_test0 = (
        torch.from_numpy(desc[:n_train]).float(),
        torch.from_numpy(desc[n_train: n_train + n_val]).float(),
        torch.from_numpy(desc[n_train + n_val:]).float(),
    )

    X_train1, X_val1, X_test1 = (
        torch.from_numpy(global_features[:n_train]).float(),
        torch.from_numpy(
            global_features[n_train: n_train + n_val]).float(),
        torch.from_numpy(global_features[n_train + n_val:]).float(),
    )

    X_train3, X_val3, X_test3 = (
        torch.from_numpy(p10b[:n_train]).float(),
        torch.from_numpy(p10b[n_train: n_train + n_val]).float(),
        torch.from_numpy(p10b[n_train + n_val:]).float(),
    )

    X_train4, X_val4, X_test4 = (
        torch.from_numpy(p11b[:n_train]).float(),
        torch.from_numpy(p11b[n_train: n_train + n_val]).float(),
        torch.from_numpy(p11b[n_train + n_val:]).float(),
    )

    Y_train, Y_val, Y_test = (
        torch.from_numpy(Target[:n_train]).float(),
        torch.from_numpy(Target[n_train: n_train + n_val]).float(),
        torch.from_numpy(Target[n_train + n_val:]).float(),
    )

    Y_train = Y_train.reshape(-1, 1)
    Y_val = Y_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return [X_train0, X_train1, X_train3, X_train4], Y_train, [X_val0, X_val1, X_val3, X_val4], Y_val, [X_test0, X_test1, X_test3, X_test4], Y_test


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience, slatm_len):
    batch_size = 32
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(
        n_train, n_val, n_test, iX, iY
    )
    print("1")

    train = torch.utils.data.TensorDataset(torch.cat(X_train, dim=1), Y_train)
    print("2")
    test = torch.utils.data.TensorDataset(torch.cat(X_test, dim=1), Y_test)
    print(3)
    valid = torch.utils.data.TensorDataset(torch.cat(X_val, dim=1), Y_val)
    print(4)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    print(5)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    print(6)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    print(7)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model = NeuralNetwork(slatm_len).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.50, patience=50, min_lr=1e-6)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    early_stopping = EarlyStopping()

    epochs = 10000
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
    ctest = open('comp-test.dat', 'w')
    for ii in range(0, len(pred)):
        ctest.write(
            '{}'.format(pred[ii]) + '{}'.format(Y_test[ii]) + '{}'.format(dtest[ii]) + '\n')
    ctest.close()

    # Save as a plot
    plt.plot(pred.cpu(), y.cpu(), '.')
    mini = min(y).item()
    maxi = max(y).item()
    temp = np.arange(mini, maxi, 0.1)
    plt.plot(temp, temp)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig('Result.png')
    plt.close()


n_train = 3258
n_val = 408
n_test = 407
patience = 100
iX, iY, slatm_len = prepare_data()

current_dir = os.getcwd()

os.chdir(current_dir + '/withdft/split/lipo/')
try:
    os.mkdir(str(n_train))
except:
    pass
os.chdir(current_dir + '/withdft/split/lipo/' + str(n_train))

model, lr, loss, mae, test_loader = fit_model_dense(
    n_train, n_val, int(n_test), iX, iY, patience, slatm_len
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
torch.save(model.state_dict(), 'model_dict.pt')

# Saving results
plotting_results(model, test_loader)
