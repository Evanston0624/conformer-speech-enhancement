import librosa
import numpy as np
import scipy.signal as ssg

def wav2spec(WavData, WinLen=512, NUM_FFT = 512, hop_length=200, NUM_FRAME = 1, returnPhase = False):
    epsilon = np.finfo(float).eps
    D = librosa.stft(y = np.float32(WavData),
                     n_fft = NUM_FFT,
                     hop_length = hop_length,
                     win_length = WinLen,
                     window = ssg.windows.hann,
                     center = True)
    return_data = np.log10(abs(D)**2 + epsilon)
    if returnPhase:
        return return_data, np.angle(D)
    else:
        return return_data #傅立葉轉換 頻譜圖

def spec2wav(Inp_1, Inp_2, WinLen = 512, NUM_FFT = 512, hop_length = 200):
    InpSpec = np.multiply(np.sqrt(10**(np.array(Inp_1))), np.exp(1j * Inp_2))
    return np.real(librosa.istft(InpSpec,
                             hop_length=hop_length,
                             win_length=WinLen,
                             window=ssg.windows.hann))