import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.io import wavfile
import numpy as np




def crossCorr(x, y):
    output = scipy.signal.correlate(x, y)
    fig = plt.figure()
    print('output is length ' + str(len(output)))
    plt.title('cross-corrleated signal')
    plt.xlabel('lag')
    plt.plot(output)

    return output


def loadSoundFile(filename):
    samplerate, data = wavfile.read(filename)
    if len(data.shape) == 2:
        data = data[:, 0]
    data = np.array(data, dtype=float)
    print(data.shape)
    return data


def mainfunc(f1, f2):
    x = loadSoundFile(f1)
    y = loadSoundFile(f2)
    return crossCorr(x, y)


def findSnarePosition(snareFilename, drumloopFileName):
    data = mainfunc(snareFilename, drumloopFileName)
    index = np.argsort(data)
    places = index[-4:]
    np.savetxt('02-snareLocation',places)
    plt.show()
    print('finished')
    return


findSnarePosition('snare.wav', 'drum_loop.wav')
