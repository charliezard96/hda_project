from scipy.io.wavfile import read
import numpy as np
import pandas as pd
from os import listdir
import time
import framing


def main():

    data_dir = "dataset\\speech_commands_v0.02"
    test = "dataset\\testing_list.txt"
    val = "dataset\\validation_list.txt"
    dest_dir = 'dTrain2'
    all_dir = np.array(listdir(data_dir))
    test_set = np.loadtxt(test, dtype=str)
    val_set = np.loadtxt(val, dtype=str)

    fs = 16000      # Sampling frequency (better time performances than extract it from the file)
    dur = 0.025     # Frame duration in ms
    step = 0.010    # Frame step size in ms
    n_mel = 26
    f_min = 300
    all_time = time.time()      # Used to evaluate time performances
    data_val = pd.DataFrame(columns=['filename', 'data', 'label'])
    data_test = pd.DataFrame(columns=['filename', 'data', 'label'])
    # Compute Mel-filterbanks coefficients only once
    mel_mat = framing.getMellFilterbanks(fs*dur, fs/2, f_min, n_mel)

    for d in all_dir[:-1]:
        all_files = np.array(listdir(data_dir+"\\"+d))
        start_time = time.time()        # Used to evaluate time performances
        data_temp = pd.DataFrame(columns=['filename', 'data', 'label'])     # Initialize sub-dataframe
        for f in all_files:
            # Read and process the sample
            src = read(data_dir+"\\"+d+"\\"+f)
            src = framing.fullFeatExtraction(np.array(src[1], dtype=float), mel_mat, fs, dur, step)
            if src.shape[0] != 100:
                continue
            # Update sub-dataframe
            temp = pd.DataFrame([[d+"/"+f, src, d]], columns=['filename', 'data', 'label'])
            data_temp = data_temp.append(temp, ignore_index=True)
        # Extract test samples
        idx_t = data_temp.index[np.in1d(data_temp['filename'], test_set)]
        data_temp_t = data_temp.iloc[idx_t]
        # Extract test samples
        idx_v = data_temp.index[np.in1d(data_temp['filename'], val_set)]
        data_temp_v = data_temp.iloc[idx_v]
        # Remove test and validation set
        data_temp = data_temp.drop(idx_t)
        data_temp = data_temp.drop(idx_v)
        data_temp.to_hdf(dest_dir+'\\'+d+'.h5', key='df', mode='w')

        # Update all the sets
        data_test = pd.concat([data_test, data_temp_t], ignore_index=False)
        data_val = pd.concat([data_val, data_temp_v], ignore_index=False)

        print(d + ": --- %s seconds ---" % (time.time() - start_time))

    # Save all the sets
    data_test.to_hdf('dTest.h5', key='df', mode='w')
    data_val.to_hdf('dVal.h5', key='df', mode='w')
    print("Dataset file creation: --- %s seconds ---" % (time.time() - all_time))
    all_time = time.time()
    # Reload train set
    print("Load dataset file: --- %s seconds ---" % (time.time() - all_time))

    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()