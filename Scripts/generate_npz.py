import numpy as np
import csv
npzdir = 'C:/raghav/Prop_pred/data/lipo/npz/npzfiles/'
file = './data/lip_filter.csv'

fileno = 0
with open(file, 'r') as file:
    next(file)
    filecontent = csv.reader(file)
    for row in filecontent:
        size = int(row[3])
        target = float(row[1])
        try:
            npzfile = np.load(npzdir + str(fileno) + '.npz')
        except:
            print("File number = ", fileno)
            print("Size = ", size, '\n')
            fileno += 1
            continue
        npzdict = dict(npzfile)
        npzdict['target'] = target
        npzdict['size'] = size
        np.savez(f"C:/raghav/Prop_pred/data/lipo/npz/{fileno}", **npzdict)
        fileno += 1
