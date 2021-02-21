import wave
from scipy.io.wavfile import read
from scipy.fftpack import dct
import numpy as np
import math
import matplotlib.pyplot as plt



def delta2Comp(mat):
    dim = mat.shape[0]
    features_mat = np.pad(mat, ((2, 2), (0, 0)))
    features_mat = features_mat[3:dim + 3, :] - features_mat[1:dim + 1, :] \
                   + 2 * (features_mat[4:dim + 4, :] - features_mat[0:dim, :])
    features_mat = features_mat / 10
    return features_mat

def framing(src, Fs, dur, step):
    f_dur = int(dur * Fs)  # Number of samples in each frame
    f_step = int(step * Fs) # Number of samples of each frame step
    tot_samples = len(src)
    # Padding
    last_idx = int(math.ceil(tot_samples / f_step - 1) * f_step)
    pad = int(f_dur - len(src[last_idx:tot_samples]))
    samples = np.pad(src, (0, pad), constant_values=np.finfo(float).eps)
    samples[samples==0] = np.finfo(float).eps
    # Frame extraction
    frame_mat = []
    for i in range(0, last_idx + 1, f_step):
        frame_mat.append(samples[i:i + f_dur])

    return np.array(frame_mat)

def dftAndPer(src):
    N = src.shape[1]
    # DFT
    window = np.hamming(N)
    frame_mat_win = np.multiply(src, window)
    dft_frame_mat = np.fft.fft(frame_mat_win)

    # Only N/2+1 samples are significant
    periodogram_mat = []
    for i in range(0, dft_frame_mat.shape[0]):
        tmp = dft_frame_mat[i, 0:math.ceil(N / 2) + 1]
        abs_tmp = np.array([abs(number) for number in tmp])
        periodogram = (abs_tmp ** 2) / N
        periodogram_mat.append(periodogram)

    return np.array(periodogram_mat)

def getMellFilterbanks(s_num, f_max, f_min=300, n_fb=26):

    f_min_mel = 1125 * math.log(1 + f_min / 700)
    f_max_mel = 1125 * math.log(1 + f_max / 700)
    n_thresh = n_fb + 2  # Number of thresholds
    mel_step = (f_max_mel - f_min_mel) / (n_thresh - 1)
    mel_vec = np.zeros(n_thresh)  # Thresholds in mel-space
    freq_vec = np.zeros(n_thresh)  # Equivalent thresholds' freq.
    freq_idx = np.zeros(n_thresh, dtype=int)

    # Compute Mel-frequencies
    for i in range(n_thresh):
        mel_vec[i] = f_min_mel + i * mel_step
        freq_vec[i] = 700 * (math.exp(mel_vec[i] / 1125) - 1)
        freq_idx[i] = int((s_num + 1) * freq_vec[i] / (f_max * 2))
    # Mel-filterbanks creation
    mel_filterbanks = np.zeros((math.ceil(s_num / 2) + 1, n_fb))
    for i in range(n_fb):
        up_step = freq_idx[i + 1] - freq_idx[i] + 1
        mel_filterbanks[freq_idx[i]:freq_idx[i + 1] + 1, i] = np.linspace(0.0, 1.0, num=up_step)
        down_step = freq_idx[i + 2] - freq_idx[i + 1] + 1
        mel_filterbanks[freq_idx[i + 1]:freq_idx[i + 2] + 1, i] = np.linspace(1.0, 0.0, num=down_step)

    return mel_filterbanks

def fullFeatExtraction(src, mel, Fs, dur=0.025, step=0.010):
    frame_mat = framing(src, Fs, dur, step)
    periodogram_mat = dftAndPer(frame_mat)
    log_mat = np.log(np.matmul(periodogram_mat, mel))
    dct_mat = dct(log_mat)
    dct_mat = dct_mat[:, 1:13]
    delta_mat = delta2Comp(dct_mat)
    delta_mat[delta_mat == 0] = np.finfo(float).eps
    delta_delta_mat = delta2Comp(delta_mat)
    delta_delta_mat[delta_delta_mat == 0] = np.finfo(float).eps
    energy = np.reshape(np.log10(np.sum(frame_mat**2, axis=1)), (-1, 1))
    delta_energy = np.reshape(np.log10(np.sum(delta_mat ** 2, axis=1)), (-1, 1))
    delta2_energy = np.reshape(np.log10(np.sum(delta_delta_mat ** 2, axis=1)), (-1, 1))
    feature_mat = np.concatenate([dct_mat, delta_mat, delta_delta_mat, energy, delta_energy, delta2_energy], axis=1)
    return feature_mat



def main():

    # Desired parameters
    f_time_dur = 0.025
    f_time_step = 0.010

    sample_rate = wave.open("dataset\\speech_commands_v0.02\\backward\\0a2b400e_nohash_0.wav", 'rb').getframerate()
    src = read("dataset\\speech_commands_v0.02\\backward\\0a2b400e_nohash_0.wav")

    # Show sound
    plt.figure(figsize=(20, 4))
    sound_plot = plt.plot(src[1])
    plt.show()

    samples = np.array(src[1], dtype=float)

    # Framing
    frame_mat = framing(samples, sample_rate, f_time_dur, f_time_step)

    # Apply DFT and compute periodogram
    periodogram_mat = dftAndPer(frame_mat)

    # Mel-filterbanks
    n_fb = 26                               # Number of filters
    f_max = sample_rate / 2                 # Nyquist freq.
    f_min = 300                             # User defined min freq.

    mel_filterbanks = getMellFilterbanks(frame_mat.shape[1], f_max, f_min, n_fb)

    # Apply mel_filterbanks and compute log(E)
    log_mat = np.log(np.matmul(periodogram_mat, mel_filterbanks))

    # Show spectrogram
    plt.figure(figsize=(20, 4))
    spectrogram = plt.imshow(np.transpose(log_mat), origin='lower')
    plt.ylabel("Mel channel")
    plt.show()

    dct_mat = dct(log_mat)
    dct_mat = dct_mat[:, 1:13]

    # Show MFCC
    plt.figure(figsize=(16, 4))
    MFCCs = plt.imshow(np.transpose(dct_mat), origin='lower')
    plt.yticks(range(0, 12, 2))
    plt.show()


    # Additional features
    # Delta coefficient
    delta_mat = delta2Comp(dct_mat)
    delta_delta_mat = delta2Comp(delta_mat)

    # Compute energies
    # Overall energy on raw frame signal
    energy = np.reshape(np.log10(np.sum(frame_mat ** 2, axis=1)), (-1, 1))
    delta_energy = np.reshape(np.log10(np.sum(delta_mat**2, axis=1)), (-1, 1))
    delta2_energy = np.reshape(np.log10(np.sum(delta_delta_mat**2, axis=1)), (-1, 1))

    feature_mat = np.concatenate([dct_mat, delta_mat, delta_delta_mat, energy, delta_energy, delta2_energy], axis=1)

    res = feature_mat - fullFeatExtraction(samples, mel_filterbanks, sample_rate, f_time_dur, f_time_step)
    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()