import wave
from scipy.io.wavfile import read
import numpy as np
import math

def main():

    #sampling rate 16kHz and 1 sec wav file
    #we need 400 sample for each frame (vedi slide 17)
    #shift di 160
    # Desired parameters
    f_time_dur = 0.025
    f_time_step = 0.010
    # Intrinsic parameters
    sample_rate = wave.open("dataset\\speech_commands_v0.02\\backward\\0a2b400e_nohash_0.wav", 'rb').getframerate()
    f_dur = f_time_dur*sample_rate      # Number of samples in each frame
    f_step = f_time_step*sample_rate    # Number of samples of each frame step

    src = read("dataset\\speech_commands_v0.02\\backward\\0a2b400e_nohash_0.wav")
    tot_samples = src[0]
    samples = np.array(src[1], dtype=float)

    # Padding
    last_idx = int(math.ceil(tot_samples/f_step-1)*f_step)
    deb = len(samples[last_idx:-1])
    pad = int(f_dur-len(samples[last_idx:tot_samples]))
    samples = np.pad(samples, (0, pad))


if __name__ == "__main__":
    main()