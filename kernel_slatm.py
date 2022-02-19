import numpy as np
import logging
import schnetpack as spk
from sklearn.preprocessing import StandardScaler
from qml.math import cho_solve
from qml.kernels import gaussian_kernel

from qml.representations import (
    get_slatm_mbtypes,
    generate_slatm,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error


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


Repre, Target = prepare_data('EAT')


def objective(params):
    global Repre
    global Target
    sigma, gamma = params
    print("sigma=", sigma)
    print("gamma=", gamma)

    n_test = 31000
    n_val = 5000

    train_set = [30000]

    # try:
    #     indices = np.arange(desc.shape[0])
    #     np.random.shuffle(indices)
    #     desc = desc[indices]
    #     dftb = dftb[indices]
    #     Target = Target[indices]
    # except Exception as e:
    #     print(e)
    #     pdb.set_trace()

    for n_train in train_set:
        X_train = np.array(Repre[:n_train])
        X_test = np.array(Repre[-n_test:])

        Y_train, Y_val, Y_test = (
            np.array(Target[:n_train]),
            np.array(Target[-n_test - n_val: -n_test]),
            np.array(Target[-n_test:]),
        )

        K = gaussian_kernel(X_train, X_train, sigma)
        K[np.diag_indices_from(K)] += gamma  # Regularizer
        alpha = cho_solve(K, Y_train)  # α=(K+λI)−1y
        Ks = gaussian_kernel(X_test, X_train, sigma)
        Y_predicted = np.dot(Ks, alpha)

        # Writing the true and predicted EAT values
        dtest = np.array(Y_test - Y_predicted)

        ctest = open('comp-test_%s.dat' % n_train, 'w')
        for ii in range(0, len(Y_test)):
            ctest.write(str(Y_test[ii]) + '\t' + str(Y_predicted[ii]) + '\t' + str(dtest[ii]) + '\n'
                        )
        ctest.close()

        MAE_PROP = float(mean_absolute_error(Y_test, Y_predicted))
        MSE_PROP = float(mean_squared_error(Y_test, Y_predicted))
        STD_PROP = float(Y_test.std())

        out2 = open('errors_test%s.dat' % n_train, 'w')
        out2.write(
            str(STD_PROP) + "\t"
            + str(MAE_PROP) + "\t"
            + str(MSE_PROP) + "\t"
            + "\n"
        )
        out2.close()

    res = np.mean(np.abs(Y_predicted - Y_test))
    print(res)
    return res


sigma, gamma = [133.55025, 1e-05]
objective([sigma, gamma])
