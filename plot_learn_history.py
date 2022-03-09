import matplotlib.pyplot as plt

# Plot the curve of learning rate

lhis = open('withdft/split/eq/20000/learning-history.dat', 'r')

lines = lhis.readlines()
x = []
y = []


for line in lines:
    epoch, lr, loss, mae = line.split()
    x.append(int(epoch))
    y.append(float(mae))

lhis = open('withdft/slatm/eq/30000/learning-history.dat', 'r')

lines = lhis.readlines()
x1 = []
y1 = []


for line in lines:
    epoch, lr, loss, mae = line.split()
    x1.append(int(epoch))
    y1.append(float(mae)*23.0621)

plt.plot(x[1500:], y[1500:], '.')
plt.plot(x1[1500:], y1[1500:], '.')

plt.ylabel("Validation MAE")
plt.xlabel("Epoch")
plt.title('Learning history for conv with size of 8000')
plt.show()
plt.close()
