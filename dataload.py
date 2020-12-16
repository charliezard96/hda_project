import wave
from scipy.io.wavfile import read
import numpy as np
import pandas as pd
from os import listdir
import time



def main():

    data_dir = "dataset\\speech_commands_v0.02"
    all_dir = np.array(listdir(data_dir))
    data = pd.DataFrame(columns=['filename', 'data', 'label'])

    all_time = time.time()
    for dir in all_dir[:-1]:
        all_files = np.array(listdir(data_dir+"\\"+dir))
        start_time = time.time()
        data_temp = pd.DataFrame(columns=['filename', 'data', 'label'])
        for f in all_files:
            src = read(data_dir+"\\"+dir+"\\"+f)
            temp = pd.DataFrame([[dir+"/"+f, np.array(src[1], dtype=float), dir]], columns=['filename', 'data', 'label'])
            data_temp = data_temp.append(temp, ignore_index=True)
        data = pd.concat([data, data_temp], ignore_index=False)
        print(dir + ": --- %s seconds ---" % (time.time() - start_time))

    print(": --- %s seconds ---" % (time.time() - all_time))
    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()