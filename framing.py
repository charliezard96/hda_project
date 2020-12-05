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
    f_dur = int(f_time_dur * sample_rate)      # Number of samples in each frame
    f_step = int(f_time_step * sample_rate)    # Number of samples of each frame step

    src = read("dataset\\speech_commands_v0.02\\backward\\0a2b400e_nohash_0.wav")
    tot_samples = src[0]
    samples = np.array(src[1], dtype=float)

    # Padding
    last_idx = int(math.ceil(tot_samples / f_step - 1) * f_step)
    pad = int(f_dur - len(samples[last_idx:tot_samples]))
    samples = np.pad(samples, (0, pad))

    # Frame extraction
    frame_mat = []
    for i in range(0, last_idx, f_step):
        frame_mat.append(samples[i:i + f_dur])

    frame_mat = np.array(frame_mat)         # Convert the list in a proper numpy array

    # DFT
    window = np.hamming(f_dur)
    frame_mat_win = np.multiply(frame_mat, window)

    dft_frame_mat = np.fft.fft(frame_mat_win)

    # Only N/2+1 samples are significant
    periodogram_mat = []
    for i in range(0, dft_frame_mat.shape[0]):
        tmp = dft_frame_mat[i, 0:math.ceil(f_dur / 2) + 1]
        abs_tmp = np.array([abs(number) for number in tmp])
        periodogram = (abs_tmp ** 2) / f_dur
        periodogram_mat.append(periodogram)

    periodogram_mat = np.array(periodogram_mat)

    # Mel-filterbanks
    n_fb = 26                               # Number of filters
    f_max = sample_rate / 2                 # Nyquist freq.
    f_min = 300                             # User defined min freq.

    # Copiato da slide 25:
    f_min_mel = 1125 * math.log(1 + f_min / 700)
    f_max_mel = 1125 * math.log(1 + f_max / 700)
    n_thresh = n_fb + 2                     # Number of thresholds
    mel_step = (f_max_mel - f_min_mel) / (n_thresh - 1)
    mel_vec = np.zeros(n_thresh)            # Thresholds in mel-space
    freq_vec = np.zeros(n_thresh)           # Equivalent thresholds' freq.

    for i in range(n_thresh):
        mel_vec[i] = f_min_mel + i * mel_step
        freq_vec[i] = 700 * (math.exp(mel_vec[i] / 1125) - 1)



    z = 1


if __name__ == "__main__":
    main()