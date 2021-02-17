import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
import timeit
from scipy.io import wavfile

def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    t = np.arange(0, length_secs, 1 / sampling_rate_Hz)
    x = amplitude * np.sin(np.pi * 2 * frequency_Hz * t + phase_radians)
    return t, x

# commenting below block to test plotting for other problems, Q1 result in folder
tt, yy1 = generateSinusoidal(1, 44100,400,0.5, 0)
# plt.plot(tt[:221],y[:221])
# plt.xlabel('time in seconds')
# plt.title('first 5ms of suinusoid in Part 2')
# plt.ylabel('amplitude')

def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    x = np.zeros(int(sampling_rate_Hz * length_secs))
    for i in range(1,2*10,2):
        t, currx = generateSinusoidal(1/(i), sampling_rate_Hz,(i) * frequency_Hz,length_secs,phase_radians)
        x += currx
    x = x * amplitude
    return t,x

tt, yy2 = generateSquare(1,44100,400,0.5,0)
# plt.plot(tt[:221],yy2[:221])
# plt.xlabel('time in seconds')
# plt.title('first 5ms of square wave')
# plt.ylabel('amplitude')
# plt.show()

def computeSepctrum(x, sample_rate_Hz):

    window_length = len(x)
    if(len(x)%2==1):
        halved = int((len(x)+1)/2)
    else:
        halved = int((len(x)/2+1))
    output = np.fft.fft(x, n=window_length)
    output = output[0:halved]
    XRe = np.real(output)
    XIm = np.imag(output)
    XAbs = np.abs(output)
    XPhase = np.angle(output)
    f = np.arange(0, halved)*sample_rate_Hz/window_length
    return XRe, XPhase, XAbs, XIm, f,

# #uncomment the block because I need to test other parts. Output is in results folder
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].set_title('Magnitude of Sine wave')
# axs[0, 0].set_xlabel('Hz')
# axs[1, 0].set_title('Phase of Sine')
# axs[0, 1].set_title('magnitude of square wave')
# axs[0, 1].set_xlabel('Hz')
# axs[1, 1].set_title('Phase of square wave')
# fig.tight_layout()
# XRe, XPhase, XAbs, XIm, f, = computeSepctrum(yy1, 44100)
# axs[0, 0].plot(f, XAbs)
# axs[1, 0].plot(XPhase)
# XRe, XPhase, XAbs, XIm, f, = computeSepctrum(yy2, 44100)
# axs[0, 1].plot(f, XAbs)
# axs[1, 1].plot(XPhase)
# plt.show()


def generateBlocks(x, sample_rate_Hz, block_size, hop_size):
    print(np.ceil(len(x)/hop_size)*hop_size-len(x))
    pad = np.zeros(int(np.ceil(len(x)/hop_size)*hop_size-len(x) + hop_size))
    new_x = np.concatenate((x, pad))
    output = np.zeros((int(np.ceil(len(x)/hop_size)), int(block_size)))
    t = np.zeros(int(np.ceil(len(x)/hop_size)))
    for i in range(int(np.ceil(len(x)/hop_size))):
        cur = new_x[i*hop_size:i*hop_size + block_size]
        output[i] = cur
        t[i] = i*hop_size/sample_rate_Hz
    X = output
    return X, t

def mySpecgram(x,  block_size, hop_size, sampling_rate_Hz, window_type):
    if window_type == 'hann':
        window = np.hanning(block_size)
    else:
        window = np.ones(block_size)
    X, t = generateBlocks(x, sampling_rate_Hz, block_size, hop_size)
    time_vector = t
    if(block_size%2==1):
        halved = int((block_size+1)/2)
    else:
        halved = int(block_size/2+1)
    magnitude_spectrogram = np.zeros((halved, X.shape[0]))
    for i in range(X.shape[0]):
        XRe, XPhase, XAbs, XIm, f, = computeSepctrum(X[i] * window, sampling_rate_Hz)
        magnitude_spectrogram[:,i] = XAbs
        freq_vector = f
    #Pxx, freqs, bins, im = plt.specgram(x, NFFT=block_size, Fs=sampling_rate_Hz, noverlap=block_size-hop_size)
    plt.pcolormesh(time_vector,freq_vector, magnitude_spectrogram)
    plt.show()
    return freq_vector, time_vector, magnitude_spectrogram
#
# a, b = generateBlocks(yy2, 44100, 2048, 1024)
out = np.fft.rfft(yy2)
freq_vector, time_vector, magnitude_spectrogram = mySpecgram(yy2, 2048, 1024, 44100, 'rect')
freq_vector, time_vector, magnitude_spectrogram = mySpecgram(yy2, 2048, 1024, 44100, 'hann')



print('finished')