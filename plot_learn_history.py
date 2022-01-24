import matplotlib.pyplot as plt

# Plot the curve of learning rate

lhis = open('only_dft/egap/10000/learning-history.dat', 'r')

lines = lhis.readlines()
x = []
y = []


for line in lines:
    epoch, lr, loss, mae = line.split()
    x.append(int(epoch))
    y.append(float(mae))

plt.plot(x[500:], y[500:], '.')
plt.xlabel("Training MAE")
plt.ylabel("Epoch")
plt.title('Learning history for conv with size of 10000')
plt.show()
plt.close()
