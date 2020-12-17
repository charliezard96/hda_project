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
    data = pd.DataFrame(columns=['filename', 'data', 'label'])
    fs = 16000
    dur = 0.025
    step = 0.010
    all_time = time.time()
    for d in all_dir[:-1]:
        all_files = np.array(listdir(data_dir+"\\"+d))
        start_time = time.time()
        data_temp = pd.DataFrame(columns=['filename', 'data', 'label'])
        #fs = wave.open(data_dir + "\\" + dir + "\\" + all_files[0], 'rb').getframerate()
        for f in all_files:
            src = read(data_dir+"\\"+d+"\\"+f)
            src = framing.framing(np.array(src[1], dtype=int), fs, dur, step)
            temp = pd.DataFrame([[d+"/"+f, src, d]], columns=['filename', 'data', 'label'])
            data_temp = data_temp.append(temp, ignore_index=True)
        data = pd.concat([data, data_temp], ignore_index=False)
        print(d + ": --- %s seconds ---" % (time.time() - start_time))

    print(": --- %s seconds ---" % (time.time() - all_time))
    prova = data['data'][0]

    data.to_csv("dataframe.csv")
    all_time = time.time()
    data = pd.read_csv("dataframe.csv")
    prova = data['data'][0]
    print(": --- %s seconds ---" % (time.time() - all_time))


    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()