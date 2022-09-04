from loading_dataset import filesPaths
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample
import librosa
import numpy as np

samplerate, signal = wavfile.read(filesPaths[6])
signal_duration = librosa.get_duration(filename=filesPaths[6])


time = np.linspace(0, signal_duration, num=len(signal), endpoint=False)

new_rate = 1000

number_of_samples = round(len(signal) * float(new_rate) / samplerate)

resampled_signal = resample(signal, number_of_samples)

new_time = np.linspace(0, signal_duration, num=len(resampled_signal), endpoint=False)  # number_of_samples

lowcut = 20.0
highcut = 250.0

nyq = 0.5 * samplerate
low = lowcut / nyq
high = highcut / nyq

b, a = butter(2, [low, high], 'bandpass', analog=False)
filtered_signal = filtfilt(b, a, resampled_signal, axis=0)

if __name__ == '__main__':
    plt.figure(figsize=(12, 12))
    sig_plot = plt.subplot(311)
    sig_plot.set_title('Original Signal')
    sig_plot.plot(time, signal)
    sig_plot.set_xlabel('Time')
    sig_plot.set_ylabel('Energy')

    resample_data = plt.subplot(312, sharex=sig_plot, sharey=sig_plot)
    resample_data.set_title('After Downsample')
    resample_data.plot(new_time, resampled_signal)
    resample_data.set_xlabel('Time')
    resample_data.set_ylabel('Energy')

    filtered_data = plt.subplot(313, sharex=sig_plot, sharey=sig_plot)
    filtered_data.set_title('After Bandpass Filter')
    filtered_data.plot(new_time, filtered_signal)
    filtered_data.set_xlabel('Time')
    filtered_data.set_ylabel('Energy')

    plt.show()
