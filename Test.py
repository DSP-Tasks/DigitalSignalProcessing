import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fft import fft, ifft

from TestTask_7.Convolution import ConvTest

"""

def add_signals(*signals):
    # Find the maximum length among all input signals
    max_length = max(len(signal) for signal in signals)

    # Initialize the summed signal as zeros
    summed_signal = np.zeros(max_length)

    # Add each signal to the summed signal
    for signal in signals:
        length = len(signal)
        summed_signal[:length] += signal

    # Plot the resulting signal
    time = np.arange(max_length)
    plt.plot(time, summed_signal)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Summed Signal")
    plt.grid(True)
    plt.show()


# Read the input signals from a file
def read_signals_from_file(file_path):
    signals = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                time, amplitude = line.strip().split()
                signals.append([float(time), float(amplitude)])
    except FileNotFoundError:
        print("File not found!")
    except ValueError:
        print("Invalid file format! Each line should contain time and amplitude separated by space.")

    return np.array(signals)


# Get the file path from the user
file_path = input("Enter the file path: ")

# Read and add the input signals
input_signals = read_signals_from_file(file_path)
if input_signals is not None:
    add_signals(input_signals)
"""
"""import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import os
import tkinter as tk


def read_file():
    file_path = filedialog.askopenfilename(title="Select text file", filetypes=[("Text files", "*.txt")])
    indices = []
    samples = []
    with open(file_path, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                indices.append(V1)
                samples.append(V2)
                line = f.readline()
            else:
                break
    return samples


s = read_file()
# Iterate over each file path
combined_signal = np.zeros_like(s)
combined_signal = np.add(combined_signal, s)
for i in range(1):
    s = read_file()
    # Extract the amplitude values from the signal
    # amplitude = s[:, 1]
    # print(amplitude)

    # Add the amplitudes to the combined signal
    combined_signal = np.add(combined_signal, s)

print(combined_signal[1:15])
# Plot the combined signal
plt.plot(combined_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Combined Signal')
plt.grid(True)
plt.show()
"""
""""
# from tkinter import *
# from tkinter import filedialog
# import numpy as np
# import matplotlib.pyplot as plt
# from comparesignals import SignalSamplesAreEqual
#
#
# def read_file():
#     file_path = filedialog.askopenfilename(title="Select text file", filetypes=[("Text files", "*.txt")])
#
#     try:
#         with open(file_path, 'r') as file:
#             contents = file.readlines()
#         return contents
#
#     except FileNotFoundError:
#         print("No file selected.")
#
#
# def signal_representation():
#     contents = read_file()
#
#     samples = []
#     for line in contents:
#         # spli each line into separate values using the split method, which splits the line on whitespace.
#         values = line.strip().split()
#         # loop through the values and convert each one to a float before appending it to the list
#         for value in values:
#             samples.append(float(value))
#
#     # print(samples)
#     # remove the first two rows
#     samples.pop(0)
#     samples.pop(1)
#
#     samples = np.array(samples)
#     # print(samples)
#
#     # Continuous representation
#     plt.figure(figsize=(10, 6))
#     plt.subplot(2, 1, 1)
#     plt.plot(samples)
#     plt.title('Continuous Representation')
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#
#     # Discrete representation
#     plt.subplot(2, 1, 2)
#     plt.stem(samples)
#     plt.title('Discrete Representation')
#     plt.xlabel('Sample Number')
#     plt.ylabel('Amplitude')
#
#     # Show the plot
#     plt.tight_layout()
#     plt.show()
#
#
# def GUI():
#     gui = Tk()
#     gui.geometry('450x400+820+300')
#     gui.resizable(False, False)
#     gui.title('Main Form')
#     gui.config(background='gray')
#     # gui.iconbitmap('D:\\Projects\\Python\\DSP\\shield.ico')
#
#     lbl = Label(gui, text='DSP Tasks', bg='yellow', font=10, width=40)
#     lbl.place(x=5, y=10)
#
#     btn = Button(gui, text='Select File', bg='pink', font=1, width=10, command=signal_representation)
#     btn.place(x=160, y=360)
#
#     gui.mainloop()
#
#
# def generat_signal(signal_type, amplitude, analogFrequency, samplingFrequency, phaseShift):
#
#     sampling_rate = 1000  # Set the sampling rate in samples per second
#     duration = 1  # Set the duration in seconds
#     t = np.arange(0, duration, 1 / sampling_rate)  # Generate time values
#
#     if signal_type == 'sin':
#         sine_wave = amplitude * np.sin(2 * np.pi * analogFrequency * t)  # Generate the sine wave
#         return sine_wave
#     else:
#         # Generate a cosine wave with a frequency of 2 Hz
#         frequency = 2  # Set the frequency in Hz
#         cosine_wave = amplitude * np.cos(2 * np.pi * analogFrequency * t)  # Generate the cosine wave
#         return cosine_wave
#
#
# # GUI()
# # test
# # SignalSamplesAreEqual("SinOutput.txt",indices,samples)
#
# x = generat_signal('sin', 3, 360, 720, 1.96349540849362)
# print(x)
"""


def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)


def IDFT(signal):
    N = len(signal)
    inverse_signal = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            angle = 2j * np.pi * k * n / N
            inverse_signal[n] += signal[k] * np.exp(angle)
        inverse_signal[n] /= N
    return inverse_signal


def dft(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            power = (-2j * np.pi * k * n) / N
            result[k] += signal[n] * np.exp(power)

    return result


def idft(amplitude, phase):
    N = len(amplitude)
    result = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            complex_value = amplitude[k] * np.exp(1j * phase[k])

            result[n] += 1 / N * complex_value * np.exp(2j * np.pi * k * n / N)

    result_idft = np.abs(result)

    return result_idft


def dft_idft(signal, inverse=False):
    N = len(signal)
    result = np.zeros(N, dtype=complex)
    sign = -2j if inverse else 2j

    for k in range(N):
        for n in range(N):
            power = (sign * np.pi * k * n) / N
            result[k] += signal[n] * np.exp(power)

        if inverse:
            result[k] /= N

    return result


def DCT(signal):
    N = len(signal)
    result = [0] * N
    for k in range(N):
        for n in range(N):
            result[k] = math.sqrt(2 / N) * (signal[n] * math.cos((np.pi / 4 * N) * (2 * n - 1) * (2 * k - 1)))

    return result


def remove_dc(signal):
    N = len(signal)
    avg = np.average(signal)
    # print(avg)
    result = [0] * N
    for i in range(N):
        result[i] = signal[i] - avg

    rounded_numbers = [round(num, 3) for num in result]

    return rounded_numbers


def smoothing(signal, points):
    result = []
    for i in range(len(signal)):
        start_index = max(0, i - points + 1)
        end_index = i + 1
        avg = sum(signal[start_index:end_index]) / (end_index - start_index)
        result.append(avg)
    return result


def sharpening(c, signal):
    def first_derivative(signal):
        y = []
        for i in range(1, len(signal)):
            y.append(signal[i] - signal[i - 1])
        return y

    def second_derivative(signal):
        y = []
        for i in range(1, len(signal) - 1):
            y.append(signal[i + 1] - 2 * signal[i] + signal[i - 1])
        return y

    if c == 'f':
        return first_derivative(signal)
    else:
        return second_derivative(signal)


def delay_signal(signal, k):
    if k >= 0:
        y = [0] * k + signal[:-k]
    else:
        y = signal[-k:] + [0] * (-k)
    return y


def fold_signal(signal):
    return signal[::-1]


def delay_folded_signal(signal, k):
    folded_signal = fold_signal(signal)
    return fold_signal(delay_signal(folded_signal, k))


def remove_dc_component(signal):
    fft_data = fft(signal)
    dc_component = np.mean(signal)
    fft_data[0] = dc_component
    y = ifft(fft_data)
    return np.real(y)


def convolution():
    indices1 = [-2, -1, 0, 1]
    samples1 = [1, 2, 1, 1]
    indices2 = [0, 1, 2, 3, 4, 5]
    samples2 = [1, -1, 0, 0, 1, 1]


    result_indices = []
    result_samples = []

    for i in range(len(indices1) + len(indices2) - 1):
        result_indices.append(indices1[0] + indices2[0] + i)  # Adjusted this line
        result_sample = 0

        for j in range(len(indices1)):
            if i - j < 0 or i - j >= len(indices2):
                continue
            result_sample += samples1[j] * samples2[i - j]

        result_samples.append(result_sample)

    print("Result Indices:", result_indices)
    print("Result Samples:", result_samples)
    ConvTest.ConvTest(result_indices, result_samples)


"""test_dft = [1, 3, 5, 7, 9, 11, 13, 15]
x = [5, 3, 8, 4, 9, 5, 3]
res1 = dft(test_dft)
amplitude = np.abs(res1)
phase = np.angle(res1)
print(amplitude)
print(phase)
"""

"""
test_idft_amp = [64, 20.9050074380220, 11.3137084989848, 8.65913760233915, 8, 8.65913760233915, 11.3137084989848, 20.9050074380220]
test_idft_phase = [0, 1.96349540849362, 2.35619449019235, 2.74889357189107, -3.14159265358979, -2.74889357189107, -2.35619449019235, -1.96349540849362]
reconstructed_signal = idft(test_idft_amp, test_idft_phase)
reconstructed_signal = [round(float(value), 1) for value in reconstructed_signal]
print(reconstructed_signal)
"""

"""test_dct_amp = [0, 1, 2, 3, 4, 5]
test_dct_phase = [50.3844170297569, 49.5528258147577, 47.5503262094184, 44.4508497187474, 40.3553390593274, 50]
y1 = DCT(test_dct_amp)
y2 = DCT(test_dct_phase)
print(y1, "\n", y2)
"""

"""
test_dc_phase = [10.4628, 7.324, 7.8834, 11.3679, 12.962, 10.4628, 7.324, 7.8834, 11.3679, 12.962, 11.3679, 12.962, 10.4628, 7.324, 7.8834, 12.962]
res = remove_dc(test_dc_phase)
print(res)
"""


signal = [1, 2, 6, 5, 4]
res = smoothing(signal, 3)
# print(res)

convolution()

"""
# Example usage
signal = np.array([0, 1, 2, 3])
# dft_result1 = dft_idft(signal, True)
dft_result = dft(signal)
print('***', dft_result)
amplitude = np.abs(dft_result)
phase = np.angle(dft_result)
print(amplitude)
print(phase)
"""
"""
idft_result = idft(signal)
# print('****', dft_result)
# print('####', dft_result)
amplitude = np.abs(idft_result)
phase = np.angle(idft_result)
print(amplitude)
print(phase)

for i, value in enumerate(dft_result):
    if i == 0:
        amplitude = np.abs(value)
        phase = np.angle(value)
    else:
        amplitude = 2 * np.abs(value)
        phase = np.angle(value)

    print(f"{amplitude:.15f}f {phase:.15f}f")
"""
"""n = 4
my_list = [6.28] * n
for i in range(n):
    my_list[i] = my_list[i] * 2

print(my_list)
"""
