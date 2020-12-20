import numpy as np
import pandas as pd
import time

def main():
    test = "dataset\\testing_list.txt"
    data = pd.read_hdf('dataframe.h5', key='df')
    f = open(test, 'r')
    test_set = np.loadtxt(test, dtype=str)
    test_data_idx = data.index[np.in1d(data['filename'], test_set)]
    test_data = data.iloc[test_data_idx]
    data = data.drop(test_data_idx)

    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()