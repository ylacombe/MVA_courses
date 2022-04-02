import librosa

def synthesis(coeff, hop_synthesis = None, win_length = 2048):
    """
    Reforms the signal from the STFT coefficient.
    coeff: STFT to be reconstruct
    hop_synthesis: new hop_length
    window: size of the FFT and of the window (same as analysis)
    return inv_s -> a float array (to be converted to np.int16 #np.int16(np.around(inv_s)))
    """
    
    inv_s = librosa.istft(coeff, hop_length = hop_synthesis, center = True)
    return inv_s#np.int16(np.around(inv_s))