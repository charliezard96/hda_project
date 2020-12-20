import wave
from scipy.io.wavfile import read
import numpy as np
import pandas as pd
from os import listdir
import time
import framing



def main():

    data_dir = "dataset\\speech_commands_v0.02"
    all_dir = np.array(listdir(data_dir))
    fs = 16000      # Sampling frequency (better time performances than extract it from the file)
    dur = 0.025     # Frame duration in ms
    step = 0.010    # Frame step size in ms
    all_time = time.time()      # Used to evaluate time performances
    data = pd.DataFrame(columns=['filename', 'data', 'label'])
    data.to_hdf('dataframe.h5', key='df', mode='w')     # Initialize dataframe file

    for d in all_dir[:-1]:
        all_files = np.array(listdir(data_dir+"\\"+d))
        start_time = time.time()        # Used to evaluate time performances
        data_temp = pd.DataFrame(columns=['filename', 'data', 'label'])
        for f in all_files:
            src = read(data_dir+"\\"+d+"\\"+f)      # Read the file
            src = framing.framing(np.array(src[1], dtype=int), fs, dur, step)       # Frame extraction
            temp = pd.DataFrame([[d+"/"+f, src, d]], columns=['filename', 'data', 'label'])     # Insert data in the sub-dataframe
            data_temp = data_temp.append(temp, ignore_index=True)       # Expand sub-dataframe
        data_temp.to_hdf('dataframe.h5', key='df', mode='a')        # Update dataset file
        #data = pd.concat([data, data_temp], ignore_index=False)
        print(d + ": --- %s seconds ---" % (time.time() - start_time))

    print("Dataset file creation: --- %s seconds ---" % (time.time() - all_time))
    all_time = time.time()
    data = pd.read_hdf('dataframe.h5', key='df')
    print("Load dataset file: --- %s seconds ---" % (time.time() - all_time))


    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()