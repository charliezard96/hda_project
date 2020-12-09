import wave
from scipy.io.wavfile import read
from scipy.signal import periodogram as comp_per
from scipy.fftpack import dct
import numpy as np
import math
import matplotlib.pyplot as plt # Visualization library
# libreria esterna per i delta, va installata manualmente prima di usarla
# from python_speech_features.base import delta


def extraction_dinamic(input):
    dim = input.shape[1]
    features_mat = np.zeros(input.shape)

    m1 = input[:, 2:dim] - input[:, 0:(dim-2)]
    m2 = 2 * (input[:, 4:dim] - input[:, 0:dim-4])
    features_mat[:, 1:dim-1] = m1
    features_mat[:, 2:dim-2] += m2
    features_mat[:, 0] = input[:, 1] + input[:, 2]
    features_mat[:, 1] += input[:, 3]
    features_mat[:, dim-2] += (-1) * input[:, dim-4]
    features_mat[:, dim-1] = (-1) * input[:, dim-2] - input[:, dim-3]
    features_mat = features_mat / 10
    return (features_mat)



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

    # Prova della fuzione, i risultati son diversi, probabilmente usa una dft diversa (senza hamming?)
    # prova = np.array(comp_per(frame_mat[0, :], fs=sample_rate)[1])


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

    # Show spectrogram
    plt.figure(figsize=(20, 4))
    spectrogram = plt.imshow(np.transpose(log_mat))
    # Mostra lo spettrogramma
    # plt.show()

    dct_mat = dct(log_mat)
    dct_mat = dct_mat[:, 1:13]

    # Overall energy on raw frame signal
    energy = np.log10(np.sum(frame_mat**2, axis=1))

    #additional features
    #delta coefficient
    delta_features_mat = np.zeros(dct_mat.shape)

    m1 = dct_mat[:, 2:12] - dct_mat[:, 0:10]
    m2 = 2 * (dct_mat[:, 4:12] - dct_mat[:, 0:8])
    delta_features_mat[:, 1:11] = m1
    delta_features_mat[:, 2:10] += m2
    delta_features_mat[:, 0] = dct_mat[:, 1] + dct_mat[:, 2]
    delta_features_mat[:, 1] += dct_mat[:, 3]
    delta_features_mat[:, 10] += (-1) * dct_mat[:, 8]
    delta_features_mat[:, 11] = (-1) * dct_mat[:, 10] - dct_mat[:, 9]
    delta_features_mat = delta_features_mat/10

    # Prova della funzione già pronta (risultati diversi)
    # prova_delta = delta(dct_mat, 2)


    # extraction_dinamics è la funzione esterna che ci fa le riche di codice sopra
    delta_feat2 = extraction_dinamic(dct_mat)
    #a è per controllare che la funzione ritorni lo stesso risultato di come avevamo fatto sopra
    a = delta_features_mat-delta_feat2

    delta_delta_mat = extraction_dinamic(delta_feat2)

    # Compute energies
    delta_energy = np.log10(np.sum(delta_features_mat**2, axis=1))
    delta2_energy = np.log10(np.sum(delta_delta_mat**2, axis=1))
    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()