from matplotlib import pyplot as plt

x = [500, 1000, 2000, 4000, 8000, 16000, 25000]
mae_min = [i * 23.0621 for i in [0.07426, 0.05495, 0.0415, 0.033567, 0.026049, 0.022558, 0.01517]]
mae_max = [i * 23.0621 for i in [0.148113206, 0.0707, 0.047019, 0.043654, 0.033311, 0.02299, 0.01517]]
mse_min = [i * 23.0621 for i in [0.02830174, 0.01302, 0.011302, 0.005692, 0.004822, 0.002247, 0.001118]]
mse_max = [i * 23.0621 for i in [0.11355, 0.0405, 0.0187, 0.021055, 0.024765, 0.00255005, 0.0016813]]

kx = [1000,2000,4000,8000,10000,20000,30000]
krr = [1.177985768,
0.97167558,
0.805988032,
0.681891594,
0.649337894,
0.544971812,
0.496483726,
]

plt.plot(x, mae_min, 'o-', label='SLATM(Reduced) + TDFTB')
plt.fill_between(x, mae_min, mae_max, alpha=0.3)
plt.plot(kx,krr, 'o-', label="KRR")
plt.legend()
plt.xlabel("Training size")
plt.ylabel("MAE (kcal/mol)")
plt.title("SLATM(Reduced) + TDFTB on Distorted(QM7X)")
plt.show()