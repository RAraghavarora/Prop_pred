import matplotlib.pyplot as plt

# Plot the curve of learning rate


def get_data(path):
    lhis = open(path, 'r')
    lines = lhis.readlines()
    x = []
    y = []

    for line in lines:
        epoch, lr, loss, mae = line.split()
        x.append(int(epoch))
        y.append(float(mae)*23.0621)

    return x, y


paths = [
    'withdft/slatm/eq/validation/128/learning-history.dat',
    'withdft/slatm/eq/validation/64/learning-history.dat',
    # 'withdft/slatm/eq/validation/32/learning-history.dat',
    'withdft/slatm/eq/validation/16/learning-history.dat',
]

labels = [
    '128',
    '64',
    # '32',
    '16',
]

i=0
for path in paths:
    x, y = get_data(path)
    plt.plot(x[50:], y[50:], '.', alpha=0.2, label=labels[i])
    i+=1


plt.ylabel("Validation MAE")
plt.xlabel("Epoch")
plt.title('Learning history')
plt.legend()
plt.show()
plt.close()
