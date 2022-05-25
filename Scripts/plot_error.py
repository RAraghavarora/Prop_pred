from matplotlib import pyplot as plt

plt.rc('font', size=21) #controls default text size
plt.rc('axes', titlesize=21) #fontsize of the title
plt.rc('axes', labelsize=21) #fontsize of the x and y labels

def dist():
    x = [500, 1000, 2000, 4000, 8000, 16000, 25000, 30000]

    mae_min = [i * 23.0621 for i in [0.61587, 0.527713358, 0.4535,
                                     0.406595, 0.34495, 0.2753, 0.153138890, 0.0948]]
    mae_max = [i * 23.0621 for i in [0.8973, 0.6244, 0.53383, 0.450876,
                                     0.406749, 0.363161057, 0.153138890, 0.0948]]
    print(mae_min)
    mse_min = [0.908451319, 0.7597, 0.636101246,
               0.925783, 0.601, 0.61544948, 0.3024495]
    mse_max = [5.09, 4.48589, 4.5, 3.77, 4.279, 1.168433, 0.3024495]

    sx = [1000, 2000, 4000, 8000, 10000, 20000, 30000]
    schnet = [i * 23.0621 for i in [1.2560445594787597,
                                    0.6645179481506348,
                                    0.5534416370391846,
                                    0.37313536167144773,
                                    0.3452569046020508,
                                    0.29169067859649656,
                                    0.2657769756317139
                                    ]]

    kx = [500, 1000, 2000, 4000, 8000, 16000]
    krr = [10.42,
           5.561,
           4.035,
           3.317,
           2.917,
           2.700]

    plt.loglog(x, mae_min, 'o-', label="SLATM(Reduced) + TDFTB")
    plt.fill_between(x, mae_min, mae_max, alpha=0.3)
    plt.loglog(kx, krr, 's-', label="KRR")
    plt.loglog(sx, schnet, 's-', label="SchNet with 3 interaction blocks")
    plt.legend()
    plt.xlabel("Training size")
    plt.ylabel("MAE (kcal/mol)")
    plt.title("Atomization Energy results on Distorted Dataset(QM7X)")

    # plt.subplot(212)
    # plt.plot(x, mse_min, 'o-')
    # plt.fill_between(x, mse_min, mse_max, alpha=0.3)
    # plt.xlabel("Training size")
    # plt.ylabel("MSE (kcal/mol)")

    plt.show()


def eq():
    x = [500, 1000, 2000, 4000, 8000, 16000, 25000, 30000]
    mae_min = [i * 23.0621 for i in [0.07426, 0.05495, 0.0415,
                                     0.033567, 0.026049, 0.022558, 0.01517, 0.013843]]
    mae_max = [i * 23.0621 for i in [0.148113206, 0.0707,
                                     0.047019, 0.043654, 0.033311, 0.02299, 0.01517, 0.013843]]
    # mse_min = [i * 23.0621 for i in [0.02830174, 0.01302,
    #                                  0.011302, 0.005692, 0.004822, 0.002247, 0.001118, 0.00094774435]]
    # mse_max = [i * 23.0621 for i in [0.11355, 0.0405,
    #                                  0.0187, 0.021055, 0.024765, 0.00255005, 0.0016813, 0.0009477443527]]

    sd_x = [500, 1000, 2000, 8000, 16000, 20000, 30000]
    mae_sd = [2.02702, 1.4177732, 1.000167,
              0.53756, 0.41186, 0.39430752, 0.3456419]

    print(mae_max)

    kx = [500, 1000, 2000, 4000, 8000, 16000]
    krr = [1.233,
           0.9660,
           0.7105,
           0.5586,
           0.4230,
           0.3091
           ]

    sx = [1000, 2000, 4000, 8000, 10000, 20000, 30000]
    schnet = [23.0621 * i for i in [0.81364,
                                    0.3064954452514648,
                                    0.11652037239074707,
                                    0.05824145793914795,
                                    0.05151303291320801,
                                    0.03021006965637207,
                                    0.024040102005004883
                                    ]]

    # plt.subplot(211)
    plt.loglog(kx, krr, 's-', label="KRR")
    plt.loglog(sx, schnet, 's-', label="SchNet with 3 interaction blocks")
    plt.loglog(x, mae_min, 'o-', label="SLATM(Reduced) + TDFTB")
    plt.fill_between(x, mae_min, mae_max, alpha=0.3, color='green')
    plt.loglog(sd_x, mae_sd, 'o:', label="SLATM(Reduced) + SDFTB")
    plt.legend()
    plt.xlabel("Training size")
    plt.ylabel("MAE (kcal/mol)")
    plt.grid()
    # plt.title("Atomization Energy results on Equilibrium Dataset(QM7X)")

    # plt.subplot(212)
    # plt.plot(x, mse_min, 'o-')
    # plt.fill_between(x, mse_min, mse_max, alpha=0.3)
    # plt.xlabel("Training size")
    # plt.ylabel("MSE (kcal/mol)")

    plt.show()


def qm9():
    x = [500, 1000, 2000, 4000, 8000, 25000, 30000, 40000]
    mae_min = [3.19089, 2.4691379, 2.054, 1.616357,
               1.323917, 1.1358, 0.94325, 0.54014]
    mae_max = [4.5214, 3.078456, 2.395478, 1.839,
               1.54422831, 1.2569, 0.94325, 0.54014]
    sdx = [8000,16000,25000,40000]               
    sd = [1.143849 ,0.84 ,0.6882253,0.51828504]
    mse_min = [i * 23.0621 for i in [0.02830174, 0.01302,
                                     0.011302, 0.005692, 0.004822, 0.002247, 0.001118, 0.00094774435]]
    mse_max = [i * 23.0621 for i in [0.11355, 0.0405,
                                     0.0187, 0.021055, 0.024765, 0.00255005, 0.0016813, 0.0009477443527]]

    print(mae_min)

    kx = [500, 1000, 2000, 4000, 8000, 16000, 40000]

    krr = [2.525,
           1.761,
           1.24,
           0.8481,
           0.5838,
           0.4287,
           0.3253,
           ]

    # sx = [1000, 2000, 4000, 8000, 10000, 20000, 30000]
    # schnet = [23.0621 * i for i in [0.81364,
    #                                 0.3064954452514648,
    #                                 0.11652037239074707,
    #                                 0.05824145793914795,
    #                                 0.05151303291320801,
    #                                 0.03021006965637207,
    #                                 0.024040102005004883
    #                                 ]]

    # plt.subplot(211)
    plt.loglog(x, mae_min, 'o-', label="SLATM(Reduced) + TDFTB")
    plt.fill_between(x, mae_min, mae_max, alpha=0.3)
    plt.loglog(kx, krr, 's-', label="KRR (SLATM + DFTB)")
    plt.loglog(sdx,sd, 'o-', label="SLATM(Reduced) + SDFTB")
    # plt.loglog(sx, schnet, 's-', label="SchNet with 3 interaction blocks")
    plt.legend()
    plt.xlabel("Training size")
    plt.ylabel("MAE (kcal/mol)")
    plt.title("Atomization Energy results on QM9")

    # plt.subplot(212)
    # plt.plot(x, mse_min, 'o-')
    # plt.fill_between(x, mse_min, mse_max, alpha=0.3)
    # plt.xlabel("Training size")
    # plt.ylabel("MSE (kcal/mol)")

    plt.show()


def egap_eq():
    x = [1000, 2000, 4000, 10000, 20000, 30000]
    mae_min = [3.502611, 3.075454, 2.34315896, 1.8516150, 1.51291835, 1.36155]

    print(mae_min)

    kx = [500, 1000, 2000, 4000, 8000, 16000, 25000]

    krr = [3.813,
           3.066,
           2.524,
           2.088,
           1.688,
           1.319,
           1.091, ]

    # sx = [1000, 2000, 4000, 8000, 10000, 20000, 30000]
    # schnet = [23.0621 * i for i in [0.81364,
    #                                 0.3064954452514648,
    #                                 0.11652037239074707,
    #                                 0.05824145793914795,
    #                                 0.05151303291320801,
    #                                 0.03021006965637207,
    #                                 0.024040102005004883
    #                                 ]]

    sx = [1000, 2000, 4000, 8000, 10000, 20000, 30000]
    schnet = [23.0621 * i for i in [0.29040336680412293,
                                    0.255615983247757,
                                    0.19224964213371276,
                                    0.1450673623085022,
                                    0.1255604820251465,
                                    0.09254979228973388,
                                    0.06862995910644532, ]]

    # plt.subplot(211)
    plt.loglog(x, mae_min, 'o-', label="SLATM(Reduced) + SDFTB")
    # plt.fill_between(x, mae_min, mae_max, alpha=0.3)
    plt.loglog(kx, krr, 's-', label="KRR (SLATM + DFTB)")
    plt.loglog(sx, schnet, 's-', label="SchNet with 3 interaction blocks")
    plt.legend()
    plt.xlabel("Training size")
    plt.ylabel("MAE (kcal/mol)")
    plt.title("HOMO-LUMO gap results on QM7X (Equilibrium)")

    # plt.subplot(212)
    # plt.plot(x, mse_min, 'o-')
    # plt.fill_between(x, mse_min, mse_max, alpha=0.3)
    # plt.xlabel("Training size")
    # plt.ylabel("MSE (kcal/mol)")

    plt.show()


def egap_dist():
    x = [500, 25000, 30000]
    mae_min = [9.13, 4.83449029, 4.65807]

    kx = [500, 1000, 2000, 4000, 8000, 16000, 25000]

    krr = [5.694,
           5.153,
           4.773,
           4.503,
           4.275,
           4.094,
           3.944, ]

    # sx = [1000, 2000, 4000, 8000, 10000, 20000, 30000]
    # schnet = [23.0621 * i for i in [0.81364,
    #                                 0.3064954452514648,
    #                                 0.11652037239074707,
    #                                 0.05824145793914795,
    #                                 0.05151303291320801,
    #                                 0.03021006965637207,
    #                                 0.024040102005004883
    #                                 ]]


    # plt.subplot(211)
    plt.loglog(x, mae_min, 'o-', label="SLATM(Reduced) + TDFTB")
    # plt.fill_between(x, mae_min, mae_max, alpha=0.3)
    plt.loglog(kx, krr, 's-', label="KRR (SLATM + DFTB)")
    # plt.loglog(sx, schnet, 's-', label="SchNet with 3 interaction blocks")
    plt.legend()
    plt.xlabel("Training size")
    plt.ylabel("MAE (kcal/mol)")
    plt.title("HOMO-LUMO gap results on QM7X (Equilibrium)")

    # plt.subplot(212)
    # plt.plot(x, mse_min, 'o-')
    # plt.fill_between(x, mse_min, mse_max, alpha=0.3)
    # plt.xlabel("Training size")
    # plt.ylabel("MSE (kcal/mol)")

    plt.show()


eq()
