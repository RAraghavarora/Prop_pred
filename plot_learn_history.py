import matplotlib.pyplot as plt

# Plot the curve of learning rate

lhis = open('only_dft/egap/30000/learning-history.dat', 'r')

lines = lhis.readlines()
x = []
y = []


for line in lines:
    epoch, lr, loss, mae = line.split()
    x.append(int(epoch))
    y.append(float(mae))

plt.plot(x[1500:], y[1500:], '.')
plt.ylabel("Validation MAE")
plt.xlabel("Epoch")
plt.title('Learning history for conv with size of 8000')
plt.show()
plt.close()
