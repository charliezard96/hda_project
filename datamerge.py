import numpy as np
import pandas as pd
import time
from os import listdir

def importDataset(data_dir = 'dTrain'):
    data_dir = 'dTrain'
    data_train = pd.DataFrame(columns=['filename', 'data', 'label'])
    all_time = time.time()
    for f in listdir(data_dir):
        start_time = time.time()
        data_train = pd.concat([data_train, pd.read_hdf(data_dir+"\\"+f)], ignore_index=False)
        print(f + ": --- %s seconds ---" % (time.time() - start_time))
    print("Load dataset file: --- %s seconds ---" % (time.time() - all_time))

    return data_train
