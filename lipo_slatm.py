import os
import numpy as np
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from qml.representations import (
    get_slatm_mbtypes,
    generate_slatm,
)


def parse_xyz_string(xyz):
    atoms = xyz.split('\n')
    xyz_list = []
    for atom in atoms:
        if len(atom.split()) != 4:
            continue
        else:
            temp = [float(i) for i in atom.split()[1:]]
            xyz_list.append(temp)
    return xyz_list


def prepare_data():
    file = '/scratch/ws/1/medranos-TUDprojects/raghav/data/lip_filter.csv'
    target = []
    smiles = []
    with open(file, 'r') as file:
        filecontent = csv.reader(file)
        for row in filecontent:
            size = row[3]
            if int(size) > 90:
                continue
            smiles.append(row[2])
            target.append(float(row[1]))

    Z = []
    xyz = []

    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        try:
            pos = mol.GetConformer().GetPositions()
        except:
            print("Unable to write ", smile)
            continue
        xyz.append(pos)
        at_nos = []
        for atom in mol.GetAtoms():
            at_nos.append(atom.GetAtomicNum())
        Z.append(at_nos)

        # xyz_block = Chem.rdmolfiles.MolToXYZBlock(mol)
        # xyz.append(parse_xyz_string(xyz_block))

    n = len(Z)
    print(n)
    idx = np.arange(n)
    np.random.seed(2314)
    idx2 = np.random.permutation(idx)

    mbtypes = get_slatm_mbtypes([Z[mol] for mol in idx2])
    slatm = [
        generate_slatm(mbtypes=mbtypes,
                       nuclear_charges=Z[mol], coordinates=xyz[mol])
        for mol in idx2
    ]
    target = [target[mol] for mol in idx2]

    return slatm, target


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)


class NeuralNetwork(nn.Module):
    def __init__(self, slatm_len):
        super(NeuralNetwork, self).__init__()

        self.lin1 = nn.Linear(slatm_len, 32)
        self.lin2 = nn.Linear(32, 8)
        self.lin4 = nn.Linear(8, 1)
        self.apply(init_weights)
        # self.flatten = nn.Flatten(-1,0)

    def forward(self, x):
        layer1 = self.lin1(x)
        layer1 = nn.functional.elu(layer1)
        layer2 = self.lin2(layer1)
        layer2 = nn.functional.elu(layer2)
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


def fit_model_dense(n_train, n_val, n_test, iX, iY, patience):
    batch_size = 32
    slatm_len = len(iX[0])
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
    model = NeuralNetwork(slatm_len).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.50, patience=100, min_lr=1e-6)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

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

    out2 = open('errors_test2.dat', 'w')
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
    ctest = open('comp-test2.dat', 'w')
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
    plt.savefig('Result2.png')
    plt.close()


n_train = 4000
n_val = 50
n_test = 5
patience = 100
iX, iY = prepare_data()

current_dir = os.getcwd()

os.chdir(current_dir + '/slatm/lipo/')
try:
    os.mkdir(str(n_train))
except:
    pass
os.chdir(current_dir + '/slatm/lipo/' + str(n_train))

slatm_len = 40604
model = NeuralNetwork(slatm_len)
model.load_state_dict(torch.load(current_dir + '/slatm/lipo/3000/model_dict.pt'))

n_train = 4000
n_val = 5
n_test = 10
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
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
                                       
plotting_results(model, train_loader)