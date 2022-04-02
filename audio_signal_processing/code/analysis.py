
import librosa

def analysis(signal, hop_analysis, win_length = 2048):
    '''
    Perform analysis.
    signal: 1D sound signal
    hop_analysis: hop_length between STFT frames
    win_length: =n_fft = size of the frames
    returns stft -> shape (frequences, times) = (1+n_fft/2, n_frames)
    '''
    #hop_length = hop_analysis
    
    stft = librosa.stft(signal, hop_length = hop_analysis, win_length = win_length, n_fft = win_length)
    

    return stft