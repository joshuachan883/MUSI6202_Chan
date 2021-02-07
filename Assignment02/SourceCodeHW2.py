import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
import timeit
from scipy.io import wavfile

# if the length of x is 200 and the length of h is 100, the length of y is 200+100-1 = 299

def myTimeConv(x, h):
    fliph = np.flip(h)
    numOfZero = x.size - 1
    zeroPaddedh = np.concatenate((np.zeros(numOfZero), fliph))
    numOfZero = h.size - 1
    zeroPaddedx = np.concatenate((x, np.zeros(numOfZero)))
    #print(zeroPaddedh.size == zeroPaddedx.size)
    output = np.zeros(x.size + h.size - 1)
    for i in range(x.size + h.size - 1):
        print(i)
        vec = zeroPaddedx * zeroPaddedh
        output[i] = np.sum(vec)
        zeroPaddedh = np.delete(zeroPaddedh, 0)
        zeroPaddedh = np.append(zeroPaddedh, 0)
    return np.flip(output)


x = np.ones(200)
rampUp = np.arange(26) / 25
rampDown = np.flip(np.arange(25) / 25)
h = np.concatenate((rampUp, rampDown))

plt.plot(myTimeConv(x, h))
plt.xlabel('time(index)')
plt.title('convolution of x and h')
plt.ylabel('value')



def CompareConv(x, h):
    time = np.zeros(2)
    start = timeit.default_timer()
    scipyResult = scipy.signal.convolve(x, h)
    stop = timeit.default_timer()
    time[0] = stop - start
    start = timeit.default_timer()
    myResult = myTimeConv(x, h)
    stop = timeit.default_timer()
    time[1] = stop - start
    print(myResult.size == scipyResult.size)
    mabs = np.sum(np.absolute(scipyResult - myResult)) / myResult.size
    m = np.sum(scipyResult - myResult) / myResult.size
    stdev = np.std(scipyResult - myResult)
    return time, stdev, m, mabs,

# loadsoundfile is a function from assignment 1
def loadSoundFile(filename):
    samplerate, data = wavfile.read(filename)
    if len(data.shape) == 2:
        data = data[:, 0]
    data = np.array(data, dtype=float)
    return data

a = np.array([2,-1,5,4])
b = np.array([1,-1,3,-2])
h = loadSoundFile('impulse-response.wav')
x = loadSoundFile('piano.wav')
time, stdev, m, mabs = CompareConv(x,h)
print('h ' + str(len(h)) + ' x ' + str(len(x)))
print('mean ' + str(m) + ' absolute mean ' + str(mabs) + ' standard deviation ' + str(stdev) + ' runtime ' + str(time))
text_file = open('Q2_results.txt','w')
text_file.write('mean ' + str(m) + ' absolute mean ' + str(mabs) + ' standard deviation ' + str(stdev) + ' runtime ' + str(time))
text_file.close()
