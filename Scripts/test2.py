from multiprocessing import Pool
import numpy as np
import itertools
import pdb
from itertools import repeat


def func(a, b):
    add = a + b
    return [add, a, b]


def compute(idx, ip):  # df will not be the input, i,j will be
    print("shape ", ip.shape)
    df = {'ds': 1, 'y': ip[:, idx[0], idx[1]]}
    return df


if __name__ == "__main__":
    i = range(5)
    j = range(4)
    paramlist = list(itertools.product(i, j))
    # out = np.zeros([5, 4])
    with Pool(4) as pool:
        # Distribute the parameter sets evenly across the cores
        ip = np.random.randn(100, 80, 50)
        res = pool.starmap(compute, zip(paramlist, repeat(ip)))

        # res = np.array(res)
        # # res = np.reshape(res, [len(paramlist), 3])
        # for i in range(len(paramlist)):
        #     out[res[i][1]][res[i][2]] = res[i][0]
        # pdb.set_trace()
