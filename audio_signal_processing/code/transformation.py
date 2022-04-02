

import numpy as np
from scipy.ndimage import median_filter
import librosa
import librosa.display
import pytsmod as tsm
import matplotlib.style
import matplotlib as mpl
#Changing the style of the plot
mpl.style.use('seaborn')
import matplotlib.pyplot as plt

from synthesis import synthesis


def HPS_TSM(stft, Fs, win_length, hop_synthesis, hop_analysis, time_length, freq_length, plot_,
    phase_lock = "rigid_identity", init_alpha = False, beta = 1):
    '''
    Compute time-scale modification based on an harmonic-percussive separation.
    Applies OLA to the percussive part and phase-vocoder on the harmonic part.
    stft: STFT from the analysis step
    Fs: frames per second (frequency of the original signal)
    hop_analysis: hop_length of the analysis phase 
    hop_synthesis: hop_length of the synthesis phase (scaling_factor = hop_synthesis/hop_analysis)
    time_length: length of the horizontal median filter
    freq_length: length of the vertical median filter
    plot_median: if True, plot the vertical and horizontal STFT's spectrogram
    phase_lock: to apply phase-locking -> either "loose","rigid_identity", "rigid_scale". Otherwise, classic phase vocoder
    init_alpha: either to apply alpha to scale the initial synthesis phase (see eq(11) from the paper)
    beta: scaling factor for rigid_scale
    '''
    stft_harm, stft_perc = harmonic_percussive_separation(stft, time_length = time_length, freq_length = freq_length, 
        plot_median = plot_, hop_analysis = hop_analysis, Fs = Fs)
    
    alpha = hop_synthesis/hop_analysis
    ex = librosa.istft(stft_perc, hop_length = hop_analysis)

    x_percussive = tsm.ola(ex, alpha)

    new_stft_harm = phase_vocoder(stft_harm, Fs, hop_synthesis, hop_analysis, phase_lock = "rigid_identity", 
            init_alpha = init_alpha, beta = beta)

    x_harmonic = synthesis(new_stft_harm, hop_synthesis = hop_synthesis, win_length = win_length)

    new_recomposed_x = x_harmonic + x_percussive

    return new_recomposed_x, x_percussive, x_harmonic


def harmonic_percussive_separation(stft, time_length = 15, freq_length = 15, plot_median = False, hop_analysis = 5, Fs = 44000):
    '''
    Separes an STFT into harmonic and percussive components.
    time_length: length of the horizontal median filter
    freq_length: length of the vertical median filter
    plot_median: if True, plot the vertical and horizontal STFT's spectrogram
    hop_analysis: parameters for the spectrogram plot
    returns harmonic_stft, percussive_stft
    '''
    #apply median filter horizontally and vertically
    Y_freq = median_filter(np.abs(stft), size=(freq_length,1),mode = 'nearest') # Y_v
    Y_time = median_filter(np.abs(stft), size=(1, time_length),mode = 'nearest') #Y_h
    if plot_median:
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(Y_freq,
                                                               ref=np.max),
                                       sr = Fs, hop_length = hop_analysis,
                                       y_axis='log', x_axis='time', ax=ax)
        ax.set_title('FREQ Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()
        
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(Y_time,
                                                               ref=np.max),
                                       sr = Fs, hop_length = hop_analysis,
                                       y_axis='log', x_axis='time', ax=ax)
        ax.set_title('TIME Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()
    
    mask_harmonic = Y_time > Y_freq
    mask_percussive = Y_freq >= Y_time
    return mask_harmonic*stft, mask_percussive*stft

def phase_vocoder(stft, Fs, hop_synthesis, hop_analysis, phase_lock = None, init_alpha = False, beta = 1):
    '''
    Compute the transformation phase of the phase vocoder.
    Implementation of the Jean Laroche's paper: "Improved Phase Vocoder Time-Scale Modification of Audio"
    stft: STFT from the analysis step
    Fs: frames per second (frequency of the original signal)
    hop_analysis: hop_length of the analysis phase 
    hop_synthesis: hop_length of the synthesis phase (scaling_factor = hop_synthesis/hop_analysis)
    phase_lock: to apply phase-locking -> either "loose","rigid_identity", "rigid_scale". Otherwise, classic phase vocoder
    init_alpha: either to apply alpha to scale the initial synthesis phase (see eq(11) from the paper)
    beta: scaling factor for rigid_scale
    '''
    if phase_lock == "rigid_identity":
        return pv_tsm_rigid_PL(stft, Fs, hop_synthesis, hop_analysis, scaled_PL = False, beta = beta, init_alpha = init_alpha)
    elif phase_lock == "rigid_scale":
        return pv_tsm_rigid_PL(stft, Fs, hop_synthesis, hop_analysis, scaled_PL = True, beta = beta, init_alpha = init_alpha)
    else:
        return pv_tsm(stft, Fs, hop_synthesis, hop_analysis, phase_lock = phase_lock, init_alpha = init_alpha)



def pv_tsm(stft, Fs, hop_synthesis, hop_analysis, phase_lock = None, init_alpha = False):

    real = np.abs(stft)
    phase = np.angle(stft)
    ph = phase.copy().T #ph : shape (time,freq)
    
    n_fft = 2 * (phase.shape[0] - 1)
    freqs = (2*np.pi*np.arange(0, phase.shape[0])/ n_fft)[np.newaxis,:]
    
    
    #INSTANTANEOUS FREQUENCY COMPUTATION
    
    delta_ph = (ph[1:,:] - (ph[:-1,:] + freqs*hop_analysis)) % (2*np.pi)
    delta_ph[delta_ph>np.pi] = delta_ph[delta_ph>np.pi] - 2*np.pi
    
    IF = freqs + np.pad(delta_ph,((0,1), (0,0)), mode='constant')/hop_analysis
    
    
    #X_MOD COMPUTATION
    phase_mod = ph.copy()
    if init_alpha:
        phase_mod[0] = hop_synthesis/hop_analysis*phase_mod[0]
    
    for j in range(1,phase_mod.shape[0]):
        phase_mod[j] = phase_mod[j-1] + IF[j-1]*hop_synthesis
        

    
    #RECONSTITUTION OF THE STFT
    phase_mod = phase_mod.T
    new_stft = real*(np.cos(phase_mod) + 1.j*np.sin(phase_mod))
    

    #LOOSE PHASE LOCK
    if phase_lock == 'loose':
        print('loose')
        #channel average : Y[channel] = Y[channel] + Y[channel-1] + Y[channel+1]  
        new_stft_copy = new_stft.copy()
        new_stft[:-1] = new_stft[:-1] + new_stft_copy[1:]  #+ (k+1)
        new_stft[1:] = new_stft[1:] + new_stft_copy[:-1]  #+ (k-1)
    return new_stft


 
def pv_tsm_rigid_PL(stft, Fs, hop_synthesis, hop_analysis, scaled_PL = False, beta = 1, init_alpha = False):
    '''
    PL: phase-locking (default: identity phase-locking)
    '''
    real = np.abs(stft)
    phase = np.angle(stft)
    
    F,T = real.shape
    
    #PEAKS LOCALISATION
    peaks = np.pad(real, ((2,2), (0,0)), 'constant') #shape (2+freq+2, time)
    peaks = (peaks[1:-3]<real)&(peaks[0:-4]<real)&(peaks[3:-1]<real)&(peaks[4:]<real)#shape (freq,time)
    
    
    #ATTRIBUTION OF PEAKS TO ALL THE FREQUENCY CHANNELS
    list_begin_end = []
    end = peaks.shape[0]
    for j in range(peaks.shape[1]):

        idx = (np.arange(len(peaks))[peaks[:,j]])
        begin_end = np.zeros((2, len(idx)))

        begin_end[0, 1:] = np.floor((idx[1:] + idx[:-1])/2)
        begin_end[1,:-1] = begin_end[0, 1:] 
        begin_end[1, -1] = end

        list_begin_end.append(begin_end.copy())
        
    #HELPER FOR THE FOLLOWING STEPS
    n_fft = 2 * (phase.shape[0] - 1)
    freqs = (2*np.pi*np.arange(0, phase.shape[0])/ n_fft)
    
    ph = phase.copy().T #ph : shape (time,freq)
    phase_mod = phase.copy().T #phase_mod: shape (time,freq)
    if init_alpha:
        phase_mod[0] = hop_synthesis/hop_analysis*phase_mod[0]
       

    for j in range(1, T):
        if scaled_PL:
            #CURRENT PEAKS
            idx = (np.arange(len(peaks))[peaks[:,j]])

            #ASSOCIATION OF THE CURRENT PEAKS CHANNELS TO THE PREVIOUS SURROUNDING PEAKS
            idx_before = (np.arange(len(peaks))[peaks[:,j-1]])
            idx_before = idx_before[(idx[:,None] >= list_begin_end[j-1][0][None,:]).sum(axis = 1) - 1]
            
        else:
            idx = (np.arange(len(peaks))[peaks[:,j]])
            idx_before = idx
            beta = 1
    
        #INSTANTANEOUS FREQUENCY COMPUTATION

        delta_ph = (ph[j,idx] - (ph[j-1,idx_before] + freqs[idx]*hop_analysis)) % (2*np.pi)
        delta_ph[delta_ph>np.pi] = delta_ph[delta_ph>np.pi] - 2*np.pi

        IF = freqs[idx] + delta_ph/hop_analysis

        
        #X_MOD COMPUTATION
        phase_mod[j, idx] = phase_mod[j-1, idx_before] + IF*hop_synthesis
        
        #now compute theta for each channel 
        thetas = phase_mod[j, idx] - beta * phase.T[j,idx]
        
        #repeat theta the right number of time
        begin_end = list_begin_end[j]
        thetas = np.repeat(thetas,(begin_end[1] - begin_end[0]).astype(int))
        
        phase_mod[j] =  thetas + beta*phase.T[j]
        
    
    #STFT RECONSTITUTION
    phase_mod = phase_mod.T
    new_stft = real*(np.cos(phase_mod) + 1.j*np.sin(phase_mod))
    
    return new_stft
