import wave
from scipy.io.wavfile import read
import numpy as np

def main():

    #sampling rate 16kHz and 1 sec wav file
    #we need 400 sample for each frame (vedi slide 17)
    #shift di 160
    a = read("dataset\\speech_commands_v0.02\\backward\\0a2b400e_nohash_0.wav")
    tot_frame = np.array(a[1], dtype=float)

    a

if __name__ == "__main__":
    main()