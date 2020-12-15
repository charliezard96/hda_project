import wave
from scipy.io.wavfile import read
import numpy as np
import pandas as pd
from os import listdir


def main():

    data_dir = "dataset\\speech_commands_v0.02"
    all_dir = np.array(listdir(data_dir))
    data = np.zeros((1, 2))
    for dir in all_dir[:-1]:
        all_files = np.array(listdir(data_dir+"\\"+dir))
        for f in all_files:
            data = np.append(data, [[f, dir]], axis=0)
        z = 1

    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()