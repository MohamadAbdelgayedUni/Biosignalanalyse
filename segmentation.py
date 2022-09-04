import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from preprocessing import filtered_signal, signal_duration
import numpy as np
from sklearn.preprocessing import normalize  # to normalize a Vector in Python


# -----------------------------------------------------------------------------
# Normalize the amplitude of a vector from -1 to 1
# -----------------------------------------------------------------------------
def get_normaled_pcg(x):
    lenght = np.shape(x)  # Get the length of the vector		
    lenght = lenght[0]  # Get the value of the length
    xMax = max(x)  # Get the maximun value of the vector
    nVec = np.zeros(lenght)  # Initializate derivate vector
    for n in range(lenght):
        nVec[n] = x[n] / xMax
    nVec = nVec - np.mean(nVec)
    nVec = np.divide(nVec, np.max(nVec))
    return nVec


# -----------------------------------------------------------------------------
# PCG Peak Detection after band filter(Test)
# -----------------------------------------------------------------------------

peak_height = 1000  # calculate the height dynamically
peaks, _ = find_peaks(filtered_signal, height=peak_height)
s = filtered_signal[peaks[0]:peaks[1]]  # Let say you want first two peaks regardless the sign
plt.plot(filtered_signal)
plt.plot(peaks, filtered_signal[peaks], "x")
# plt.plot(np.zeros_like(filtered_signal), "--", color="red")
plt.show()


# -----------------------------------------------------------------------------
# Peak Detection Process
# -----------------------------------------------------------------------------
def PDP(Xf, samplerate):
    timeCut = samplerate * 0.25  # Time to count another pulse
    threshold = 0.4  # Amplitude threshold
    Xf = get_normaled_pcg(Xf)  # Normalize signal
    # Xf = normalize(Xf.reshape(1, -1))

    # -----------------------------------------------------------------------------
    # Derivate of an input signal as y[n]= x[n+1]- x[n-1]  for all values where the signal is positive
    # -----------------------------------------------------------------------------
    lenght = Xf.shape[0]  # Get the length of the vector
    y = np.zeros(lenght)  # Initializate derivate vector
    for i in range(lenght - 1):
        if Xf[i] > 0:
            y[i] = Xf[i - 1] - Xf[i]
    dX = get_normaled_pcg(y)  # Vector Normalizing
    # dX = normalize(y.reshape(1, -1))

    size = np.shape(Xf)  # Rank or dimension of the array
    fil = size[0]  # Number of rows

    positive = np.zeros((1, fil + 1))  # Initializating Vector
    positive = positive[0]  # Getting the Vector

    points = np.zeros((1, fil))  # Initializating the Peak Points Vector
    points = points[0]  # Getting the point vector

    points1 = np.zeros((1, fil))  # Initializating the Peak Points Vector
    points1 = points1[0]  # Getting the point vector

    # -----------------------------------------------------------------------------
    # FIRST! having the positives values of the slope as 1
    # And the negative values of the slope as 0
    # -----------------------------------------------------------------------------
    for i in range(0, fil):
        if Xf[i] > 0:
            if dX[i] > 0:
                positive[i] = Xf[i]
            else:
                positive[i] = 0
    # -----------------------------------------------------------------------------
    # SECOND! a peak will be found when the ith value is equal to 1 &&
    # the ith+1 is equal to 0
    # -----------------------------------------------------------------------------
    for i in range(0, fil):
        if positive[i] == Xf[i] and positive[i + 1] == 0:
            points[i] = Xf[i]
        else:
            points[i] = 0

    # -----------------------------------------------------------------------------
    # THIRD! Define a minimun Peak Height
    # -----------------------------------------------------------------------------
    p = 0
    for i in range(0, fil):
        if Xf[i] > threshold and p == 0:
            p = i
            points1[i] = Xf[i]
        else:
            points1[i] = 0
            if p + timeCut < i:
                p = 0

    return points1, points, positive[0:(len(positive) - 1)]


# -----------------------------------------------------------------------------
# Segmenting in an specific frequency band after after
# -----------------------------------------------------------------------------
pcgPeaks, peaks, allpeaks = PDP(filtered_signal, 1000)

# -----------------------------------------------------------------------------
# Time processing
# -----------------------------------------------------------------------------
dT = 0.4  # Avg Diastole time in ms
timeV = [0]
pointV = [0]
segmV = np.zeros(len(filtered_signal))

for i in range(len(pcgPeaks) - 1):
    if pcgPeaks[i] > 0.4:
        # Time of detected PCG_Peaks
        timeV.append(i / 1000)
        pointV.append(i)
        if timeV[-1] - timeV[-2] < dT:
            segmV[pointV[-2]:pointV[-1]] = 0.6
            print(f'Found: Distolic {pcgPeaks[i]} on [{i}]')
        else:
            segmV[pointV[-2]:pointV[-1]] = 0.4
            print(f'Found: Systolic {pcgPeaks[i]} on [{i}]')


def get_vektor_time(pcg, dur):
    return np.linspace(0, dur, np.size(pcg))


# -----------------------------------------------------------------------------
# Plotting Time Signal
# -----------------------------------------------------------------------------
plt.figure("Fig.5 Segmentation Heart Beats")
plt.title('Segmentation Heart Beats')

plt.subplot(3, 1, 1)
plt.title("S1 to S2")
plt.xlabel("Time")
plt.ylabel('Amplitude')
plt.plot(get_vektor_time(filtered_signal, signal_duration), get_normaled_pcg(filtered_signal),
         get_vektor_time(filtered_signal, signal_duration), pcgPeaks)

plt.subplot(3, 1, 2)
plt.title("Systolic & Distolic")
plt.xlabel("Time")
plt.ylabel('Amplitude')
plt.plot(get_vektor_time(filtered_signal, signal_duration), get_normaled_pcg(filtered_signal),
         get_vektor_time(filtered_signal, signal_duration), peaks)

plt.subplot(3, 1, 3)
plt.title("All Peaks")
plt.xlabel("Time")
plt.ylabel('Amplitude')
plt.plot(get_vektor_time(filtered_signal, signal_duration), get_normaled_pcg(filtered_signal),
         get_vektor_time(filtered_signal, signal_duration), allpeaks)

plt.show()
