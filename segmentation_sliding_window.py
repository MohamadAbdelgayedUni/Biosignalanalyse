import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from preprocessing import filtered_signal, signal_duration, new_time
import numpy as np


# -----------------------------------------------------------------------------
# new Try to segmentation with sliding window
# -----------------------------------------------------------------------------


# print(len(new_time))
i = 0

all_peaks = []
time_of_peaks = []

while i <= len(new_time):
    start = i
    end = i + 120

    peak_height = 1000  # calculate the height dynamically
    threshold = 0.4

    window = filtered_signal[start:end]
    time_of_window = new_time[start:end]

    peaks, _ = find_peaks(window, height=peak_height, threshold=threshold, distance=120)

    # print(peaks)
    arr = np.array(window)
    time_arr = np.array(time_of_window)

    all_peaks.extend(arr[peaks])  # the values of all peaks
    time_of_peaks.extend(time_arr[peaks])

    plt.plot(window)
    plt.plot(peaks, window[peaks], "x")
    plt.ylim(-15000, 15000)
    # plt.plot(np.zeros_like(filtered_signal), "--", color="red")
    plt.show()

    i = end

# print(all_peaks)
# print(time_of_peaks)

