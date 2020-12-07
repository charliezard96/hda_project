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
    for i in range(0, last_idx + 1, f_step):
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

    # Rielaborato da slide 25:
    f_min_mel = 1125 * math.log(1 + f_min / 700)
    f_max_mel = 1125 * math.log(1 + f_max / 700)
    n_thresh = n_fb + 2                     # Number of thresholds
    mel_step = (f_max_mel - f_min_mel) / (n_thresh - 1)
    mel_vec = np.zeros(n_thresh)            # Thresholds in mel-space
    freq_vec = np.zeros(n_thresh)           # Equivalent thresholds' freq.
    freq_idx = np.zeros(n_thresh, dtype=int)

    for i in range(n_thresh):
        mel_vec[i] = f_min_mel + i * mel_step
        freq_vec[i] = 700 * (math.exp(mel_vec[i] / 1125) - 1)
        # Non chiaro cosa "nDFT", assumo il numero di samples della dft (prima del periodogram)
        # NB:   se nDFT=Numero colone periodogram gli indici arrivano massimo a 100, quindi i restandi 100 sarebbero inutili, per questo ho scelto il numero dei sample della dft
        #       Inoltre con il "+1" arrivano a 200, mentre senza solo a 199, quindi ho lasciato l'ho lasciato, anche se non ho propriamente capito a cosa serva
        freq_idx[i] = int((dft_frame_mat.shape[1] + 1) * freq_vec[i] / sample_rate) # Indici delle mel-freq nelle righe della matrice del periodogram

    # Creo ora una matrice le cui righe siano i valori effettivi di ogni filtro

    mel_filterbanks = np.zeros((periodogram_mat.shape[1], n_fb))
    for i in range(n_fb):
        up_step = freq_idx[i + 1] - freq_idx[i] + 1
        mel_filterbanks[freq_idx[i]:freq_idx[i + 1] + 1, i] = np.linspace(0.0, 1.0, num=up_step)
        down_step = freq_idx[i + 2] - freq_idx[i + 1] + 1
        mel_filterbanks[freq_idx[i + 1]:freq_idx[i + 2] + 1, i] = np.linspace(1.0, 0.0, num=down_step)

    # Ogni colonna è un filterbank, quindi basta un prodotto tra matrici pe applicare i filti e sommare i valori
    # Apply mel_filterbanks and compute log(E)
    log_mat = np.log(np.matmul(periodogram_mat, mel_filterbanks))

    z = 1       # Linea di debugging


if __name__ == "__main__":
    main()