import matplotlib.style
import matplotlib as mpl
import matplotlib.pyplot as plt
#Changing the style of the plot
mpl.style.use('seaborn')

import os, sys, wave, struct
import numpy as np

def load_sound(file):
    '''
    load wave sound
    '''
    return wave.open(file, 'rb')


def plot_sound(data, times, name='default_name', save=False):
    '''
    plot sound
    '''
    plt.figure(figsize=(30, 4))
    plt.fill_between(times, data)
    plt.xlim(times[0], times[-1])
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    if save:
        plt.savefig(name+'.png', dpi=100)
    plt.show()

def play_sound(file, chunk = 1024):
    """
    Script from PyAudio doc
    BEWARE: not used, neither tried here
    """
    wf = load_sound(file)
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(chunk)

    while data:
        stream.write(data)
        data = wf.readframes(chunk )

    stream.stop_stream()
    stream.close()
    p.terminate()
    

def period(x, Fs, Pmin=1/300, Pmax=1/80, seuil=0.7) :
    '''
    Taken from a Télécom Paris' audio signal processing course
    x: short 1D audio signal
    Fs: frames per second
    returns (P,voiced) where P period of the signal if it is voiced, voiced a boolean indicating if voiced.
    '''
    # [P,voiced] = period(x,Fs,Pmin,Pmax,seuil);
    # If voiced = 1, P is the period signal x expressed in number of samples
    # If voiced = 0, P is equal to 10ms.Fs

    x = x - np.mean(x)
    N = len(x)

    Nmin = np.ceil(Pmin*Fs).astype(int)
    Nmax = 1 + np.floor(Pmax*Fs).astype(int)
    Nmax = np.min([Nmax,N])

    Nfft = int(2**np.ceil(np.log2(abs((2*N-1)))))
    X = np.fft.fft(x, n=Nfft)
    S = X * np.conj(X) / N
    r = np.real(np.fft.ifft(S))

    rmax = np.max(r[Nmin:Nmax])
    I = np.argmax(r[Nmin:Nmax])
    P = I+Nmin
    corr = (rmax/r[0]) * (N/(N-P))
    voiced = corr > seuil
    if not(voiced):
        P = np.round(10e-3*Fs)

    return P,voiced