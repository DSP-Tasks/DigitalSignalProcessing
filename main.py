import math
import cmath
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt

from TestTask_6.Derivative.DerivativeSignal import DerivativeSignal
from TestTask_6.Shifting_and_Folding.Shift_Fold_Signal import Shift_Fold_Signal
from TestTask_7.Convolution import ConvTest
from comparesignals import SignalSamplesAreEqual
import comparesignal2
from QuanTest1 import QuantizationTest1
from QuanTest2 import QuantizationTest2
from signalcompare import SignalComapreAmplitude, SignalComaprePhaseShift


def read_file():
    file_path = filedialog.askopenfilename(title="Select text file", filetypes=[("Text files", "*.txt")])

    try:
        indices = []
        samples = []
        with open(file_path, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # Process line
                L = line.strip()
                L = L.replace('f', '')
                if len(L.split(' ')) == 2:
                    L = L.split(' ')
                    V1 = float(L[0])
                    V2 = float(L[1])
                    indices.append(V1)
                    samples.append(V2)
                    line = f.readline()

                elif len(L.split(',')) == 2:
                    L = L.split(',')
                    V1 = float(L[0])
                    V2 = float(L[1])
                    indices.append(V1)
                    samples.append(V2)
                    line = f.readline()

                else:
                    break
        return indices, samples
    except FileNotFoundError:
        print("No file selected.")


def read_text():
    file_path = filedialog.askopenfilename(title="Select text file", filetypes=[("Text files", "*.txt")])

    try:
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
                L = line.replace('f', ' ')
                if len(L.split(' ')) == 2:
                    L = line.split('')
                    V1 = float(L[0])
                    V2 = float(L[1])
                    indices.append(V1)
                    samples.append(V2)
                    line = f.readline()
                else:
                    break
        return indices, samples
    except FileNotFoundError:
        print("No file selected.")


def generate_signal_from_parameters(amplitude, phase_shift, analog_frequency, sampling_frequency, is_sine=True):
    t = np.arange(0, 1, 1 / sampling_frequency)
    if is_sine:
        signal = amplitude * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
    else:
        signal = amplitude * np.cos(2 * np.pi * analog_frequency * t + phase_shift)
    return t, signal


def generate_sine_signal():
    generate_signal_gui(True)


def generate_cosine_signal():
    generate_signal_gui(False)


def generate_signal_gui(is_sine=True):
    # Create a new dialog to input signal parameters
    param_dialog = Tk()
    param_dialog.title('Signal Parameters')
    param_dialog.geometry('450x400+820+300')

    lbl = Label(param_dialog, text='Signal Parameters', font=("Arial", 12))
    lbl.pack()

    amplitude_label = Label(param_dialog, text='Amplitude:')
    amplitude_label.pack()
    amplitude_entry = Entry(param_dialog)
    amplitude_entry.pack()

    phase_shift_label = Label(param_dialog, text='Phase Shift (radians):')
    phase_shift_label.pack()
    phase_shift_entry = Entry(param_dialog)
    phase_shift_entry.pack()

    analog_frequency_label = Label(param_dialog, text='Analog Frequency (Hz):')
    analog_frequency_label.pack()
    analog_frequency_entry = Entry(param_dialog)
    analog_frequency_entry.pack()

    sampling_frequency_label = Label(param_dialog, text='Sampling Frequency (Hz):')
    sampling_frequency_label.pack()
    sampling_frequency_entry = Entry(param_dialog)
    sampling_frequency_entry.pack()

    def plot_generated_signal():
        amplitude = float(amplitude_entry.get())
        phase_shift = float(phase_shift_entry.get())
        analog_frequency = float(analog_frequency_entry.get())
        sampling_frequency = float(sampling_frequency_entry.get())

        t, signal = generate_signal_from_parameters(amplitude, phase_shift, analog_frequency, sampling_frequency,
                                                    is_sine)

        # Call SignalSamplesAreEqual function for comparing generated signals
        signal_file_name = 'TestTask_1\SinOutput.txt' if is_sine else 'TestTask_1\CosOutput.txt'
        SignalSamplesAreEqual(signal_file_name, list(range(len(signal))), signal)

        param_dialog.destroy()
        plot_signal(t, signal, is_sine)

    generate_button = Button(param_dialog, text='Generate Signal', command=plot_generated_signal)
    generate_button.pack()

    param_dialog.mainloop()


def signal_representation():
    indices, samples = read_file()

    samples = np.array(samples)
    # print(samples)
    # Continuous representation
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(samples)
    plt.title('Continuous Representation')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Discrete representation
    plt.subplot(2, 1, 2)
    plt.stem(samples)
    plt.title('Discrete Representation')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def plot_signal(t, signal, is_sine=True):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Continuous Representation')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    if is_sine:
        signal_type = 'Sine'
    else:
        signal_type = 'Cosine'

    plt.suptitle(f'{signal_type} Waveform')

    # Discrete representation
    plt.subplot(2, 1, 2)
    plt.stem(t, signal, linefmt='-g', markerfmt='go', basefmt=' ')
    plt.title('Discrete Representation')
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def plotting(indices, signal, title):
    plt.figure(figsize=(10, 6))
    plt.plot(indices, signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    plt.show()


def add_signals():
    input_gui = Tk()
    input_gui.geometry('350x300+820+300')
    input_gui.resizable(False, False)
    input_gui.title('Addition')
    num_of_signal_lbl = Label(input_gui, text='Number of Signals:')
    num_of_signal_lbl.pack()
    num_of_signal_entry = Entry(input_gui)
    num_of_signal_entry.pack()

    def gen_added_signal():
        inupt_user = num_of_signal_entry.get()
        try:
            num = int(inupt_user)
            if num == 0:
                messagebox.showwarning(title="Warning", message="Enter two signal at least")
            else:
                indices, samples = read_file()
                # Iterate over each file path
                combined_signals = np.zeros_like(samples)
                combined_signals = np.add(combined_signals, samples)
                for i in range(num - 1):
                    indices, samples = read_file()
                    combined_signals = np.add(combined_signals, samples)

                print(combined_signals)
                # Plot the combined signal
                plotting(indices, combined_signals, 'Combined Signal')

        except ValueError:
            messagebox.showwarning(title="Warning", message="Enter the number of signals")

        input_gui.destroy()

    generate_button = Button(input_gui, text='Generate Signal', command=gen_added_signal)
    generate_button.pack()
    input_gui.mainloop()


def sub_signals():
    indices, samples = read_file()
    indices1, samples2 = read_file()
    # Iterate over each file path
    subtracted_signals = np.array(samples) - np.array(samples2)

    print(subtracted_signals)
    # Plot the subtracted_signals
    plotting(indices, subtracted_signals, 'subtracted Signal')


def multiply_signal():
    input_gui = Tk()
    input_gui.geometry('350x300+820+300')
    input_gui.resizable(False, False)
    input_gui.title('Multiplication')
    num_of_signal_lbl = Label(input_gui, text='Enter a number:')
    num_of_signal_lbl.pack()
    num_of_signal_entry = Entry(input_gui)
    num_of_signal_entry.pack()

    def gen_multiplied_signal():
        inupt_user = num_of_signal_entry.get()
        try:
            num = int(inupt_user)
            indices, samples = read_file()
            # Multiply each value in the list by the constant
            multiplied_signal = [n * num for n in samples]

            print(multiplied_signal)
            # Plot the multiplied_signal
            plotting(indices, multiplied_signal, 'Multiplied Signal')

        except ValueError:
            messagebox.showwarning(title="Warning", message="Enter the number of signals")

        input_gui.destroy()

    generate_button = Button(input_gui, text='Generate Signal', command=gen_multiplied_signal)
    generate_button.pack()
    input_gui.mainloop()


def squaring_signal():
    indices, samples = read_file()

    for i in range(len(samples)):
        samples[i] = samples[i] ** 2

    print(samples)
    # Plot the multiplied_signal
    plotting(indices, samples, 'Squaring Signal')


def normalize_signal():
    input_gui = Tk()
    input_gui.geometry('250x200+820+300')
    input_gui.resizable(False, False)
    input_gui.title('Normalization')

    def normalize(zero=True):
        indices, samples = read_file()
        max_value = max(samples)
        min_value = min(samples)
        if zero:
            # Normalize each value to the range 0 to 1
            normalized_list = [(x - min_value) / (max_value - min_value) for x in samples]

            print(normalized_list)
            plotting(indices, normalized_list, 'Normalized Signal')
        else:
            # Normalize each value to the range -1 to 1
            normalized_list = [2 * ((x - min_value) / (max_value - min_value)) - 1 for x in samples]

            print(normalized_list)
            plotting(indices, normalized_list, 'Normalized Signal')

    def zero():
        normalize(True)
        input_gui.destroy()

    def one():
        normalize(False)
        input_gui.destroy()

    choise_lbl = Label(input_gui, text='Normalize from 0 to 1', font=("Arial", 12), width=40)
    choise_lbl.pack()
    generate_button = Button(input_gui, text='Generate Signal', command=zero)
    generate_button.pack()

    choise_lbl = Label(input_gui, text='Normalize from -1 to 1', font=("Arial", 12), width=40)
    choise_lbl.pack()
    generate_button = Button(input_gui, text='Generate Signal', command=one)
    generate_button.pack()

    input_gui.mainloop()


def shift_signal():
    input_gui = Tk()
    input_gui.geometry('350x300+820+300')
    input_gui.resizable(False, False)
    input_gui.title('Shifting')

    shift_label = Label(input_gui, text='Enter the shift constant:')
    shift_label.pack()

    shift_entry = Entry(input_gui)
    shift_entry.pack()

    def shift_signal_operation():
        try:
            shift_value = float(shift_entry.get())
            indices, samples = read_file()

            # Shift the signal by adding the shift constant
            shifted_signal = [x + shift_value for x in indices]

            print(shifted_signal)
            # Plot the shifted signal
            plotting(shifted_signal, samples, f'Shifted Signal ({shift_value})')

        except ValueError:
            messagebox.showwarning(title="Warning", message="Invalid input. Please enter a numeric value.")

        input_gui.destroy()

    shift_button = Button(input_gui, text='Shift Signal', command=shift_signal_operation)
    shift_button.pack()

    input_gui.mainloop()


def accumulate_signal():
    indices, samples = read_file()

    # Initialize an array to store the accumulated signal
    accumulated_signal = [0] * len(samples)

    for n in range(len(samples)):
        # Approximate the accumulation by summing from -infinity to n
        accumulated_signal[n] = sum(samples[0:n + 1])

    print(accumulated_signal)
    # Plot the accumulated signal
    plotting(indices, accumulated_signal, 'Accumulated Signal')


def quantize_signal():
    input_gui = Tk()
    input_gui.geometry('350x300+820+300')
    input_gui.resizable(False, False)
    input_gui.title('Quantiztion')

    level_label = Label(input_gui, text='Enter the Number of Levels:')
    level_label.pack()

    level_entry = Entry(input_gui)
    level_entry.pack()

    bits_label = Label(input_gui, text='Enter the Number of Bits:')
    bits_label.pack()

    bits_entry = Entry(input_gui)
    bits_entry.pack()

    def quantize_signal_operation():
        try:
            if level_entry.get() == "":
                levels = int(math.pow(2, int(bits_entry.get())))
                bits = int(bits_entry.get())
            else:
                levels = int(level_entry.get())
                bits = math.log2(levels)

            indices, samples = read_file()
            max_val = np.max(samples)
            min_val = np.min(samples)

            delta = (max_val - min_val) / levels
            ranges = [min_val]
            quantized_signal = list(samples)
            quantization_error = []
            level = []
            encoded_signal = []
            midpoint_list = [min_val + (0.5 * delta)]
            for i in range(1, levels):
                ranges.append(min_val + (i * delta))
                midpoint_list.append(ranges[i] + (0.5 * delta))

            for j in range(len(samples)):
                if samples[j] == min_val:
                    quantized_signal[j] = midpoint_list[0]
                    level.append(1)
                    b = (str(bin(0))[2:]).zfill(int(bits))
                    encoded_signal.append(b)
                    continue
                if samples[j] == max_val:
                    quantized_signal[j] = midpoint_list[levels - 1]
                    level.append(levels)
                    b = (str(bin(levels - 1))[2:]).zfill(int(bits))
                    encoded_signal.append(b)
                    continue
                if samples[j] >= ranges[levels - 1]:
                    quantized_signal[j] = midpoint_list[levels - 1]
                    level.append(levels)
                    b = (str(bin(levels - 1))[2:]).zfill(int(bits))
                    encoded_signal.append(b)
                    continue
                for k in range(levels - 1):
                    if ranges[k] <= samples[j] < ranges[k + 1]:
                        quantized_signal[j] = midpoint_list[k]
                        level.append(k + 1)
                        b = (str(bin(k))[2:]).zfill(int(bits))
                        encoded_signal.append(b)
                        break

            for i in range(len(samples)):
                quantization_error.append(quantized_signal[i] - samples[i])

            mse = 0
            for i in range(len(quantization_error)):
                mse += quantization_error[i]

            mse /= len(quantization_error)

            # Display the results
            # print(ranges)
            # print(midpoint_list)
            print(level)
            print("Quantized Signal:")
            print(quantized_signal)
            plotting(indices, quantized_signal, "quantized signal")
            print("Quantization Error:")
            print(quantization_error)
            # plotting(indices, quantization_error, "quantization_error")
            print("Encoded Signal (Levels):")
            print(encoded_signal)
            print("mean square error:")
            print(mse)
            # plotting(indices, encoded_signal, "encoded_signal")
            # QuantizationTest1("TestTask_3\Test 1\Quan1_Out.txt",encoded_signal,quantized_signal)
            QuantizationTest2("TestTask_3\Test 2\Quan2_Out.txt", level, encoded_signal, quantized_signal, quantization_error)
        except ValueError:
            messagebox.showwarning(title="Warning", message="Invalid input. Please enter a numeric value.")

        input_gui.destroy()

    quantize_button = Button(input_gui, text='quantize Signal', command=quantize_signal_operation)
    quantize_button.pack()

    input_gui.mainloop()


frequency_components = None
sampling_frequency = None


def dft(signal):
    N = len(signal)
    result = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            power = (-2j * np.pi * k * n) / N
            result[k] += signal[n] * np.exp(power)

    return result


def apply_fourier_transform(sampling_freq):
    global frequency_components, sampling_frequency
    indices, samples = read_file()

    sampling_frequency = sampling_freq

    dft_result = dft(samples)
    amplitude = np.abs(dft_result)
    phase = np.angle(dft_result)

    indices1, samples1 = read_text()
    print(amplitude)
    print(phase)
    print(SignalComapreAmplitude(samples1, amplitude))
    print(SignalComaprePhaseShift(indices1, phase))

    fundamental_freq = [i + 1 * (2 * np.pi * sampling_frequency / len(samples)) for i in range(len(samples))]
    # print(fundamental_freq)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.stem(fundamental_freq, amplitude)
    plt.title('Frequency versus Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.stem(fundamental_freq, phase)
    plt.title('Frequency versus Phase')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians')

    plt.tight_layout()
    plt.show()

    # Save frequency components in polar form
    frequency_components = list(zip(amplitude, phase))


def modify_amplitude_phase():
    global frequency_components, sampling_frequency
    if frequency_components is None:
        messagebox.showwarning(title="Warning", message="Please apply Fourier Transform first")
        return

    index = input("Enter index :")
    amplitude_modifier = float(input("Enter amplitude modification factor: "))
    phase_modifier = float(input("Enter phase modification (radians): "))

    modified_amplitude = [amp for amp, _ in frequency_components]
    modified_phase = [ph for _, ph in frequency_components]

    modified_amplitude[int(index)] *= amplitude_modifier
    modified_phase[int(index)] += phase_modifier

    reconstructed_signal = idft(modified_amplitude, modified_phase)

    print(modified_amplitude)
    # Calculate frequency resolution for the modified components
    freq_resolution = sampling_frequency / len(frequency_components)

    plt.figure(figsize=(10, 6))
    # plt.subplot(2, 1, 1)
    # frequencies = np.arange(0, sampling_frequency, freq_resolution)
    plt.plot(reconstructed_signal)
    plt.title('reconstruced signal')
    plt.xlabel('time')
    plt.ylabel('Amplitude')

    # plt.figure(figsize=(10, 6))
    # plt.subplot(2, 1, 1)
    # frequencies = np.arange(0, sampling_frequency, freq_resolution)
    # plt.stem(frequencies, modified_amplitude)
    # plt.title('Modified Frequency versus Amplitude')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    #
    # plt.subplot(2, 1, 2)
    # plt.stem(frequencies, modified_phase)
    # plt.title('Modified Frequency versus Phase')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Phase (radians')

    plt.tight_layout()
    plt.show()

    # Update frequency components with modifications
    frequency_components = list(zip(modified_amplitude, modified_phase))


def save_frequency_components(dft_applied=True):
    global frequency_components
    if frequency_components is None:
        print("No frequency components to save.")
        return

    file_name = "frequency_components.txt"
    with open(file_name, "w") as file:

        file.write("0\n")

        if dft_applied:
            file.write("1\n")
        else:
            file.write("0\n")

        file.write(f"{len(frequency_components)}\n")

        for amp, ph in frequency_components:
            file.write(f"{amp} {ph}\n")

    print(f"Frequency components saved to {file_name}")


def idft(amplitude, phase):
    N = len(amplitude)
    result = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            complex_value = amplitude[k] * np.exp(1j * phase[k])

            result[n] += 1 / N * complex_value * np.exp(2j * np.pi * k * n / N)

    result_idft = result.real

    return result_idft


def read_and_reconstruct():
    amplitude, phase = read_file()
    if not amplitude or not phase:
        print("No valid data in the file.")
        return

    reconstructed_signal = idft(amplitude, phase)
    reconstructed_signal = [round(float(value), 1) for value in reconstructed_signal]
    print(reconstructed_signal)
    indices2, samples2 = read_file()
    # print(indices2)
    print(samples2)
    print(SignalComapreAmplitude(samples2, reconstructed_signal))

    plt.figure(figsize=(10, 6))
    plt.plot(reconstructed_signal)
    plt.title('Time Domain Signal (Real Part)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def dct():
    indices, samples = read_file()
    N = len(samples)
    result = [0] * N
    for k in range(N):
        sum_val = 0
        for n in range(N):
            sum_val += samples[n] * math.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))

        result[k] = math.sqrt(2 / N) * sum_val

    file_name = 'TestTask_5\DCT\DCT_output.txt'
    comparesignal2.SignalSamplesAreEqual(file_name, result)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(indices, samples)
    plt.title('Signal before Computing DCT')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(indices, result)
    plt.title('Signal after Computing DCT')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    # m = int(input("Enter the number of coefficients to save: "))
    # file_name = f'DCT_output_{m}_coefficients.txt'
    # with open(file_name, 'w') as file:
    #     file.write("0\n")
    #     file.write("1\n")
    #     file.write(f'{m}\n')
    #     for i in range(m):
    #         file.write(f'{0} {result[i]:.5f}\n')


def remove_dc():
    indices, samples = read_file()
    N = len(samples)
    avg = np.average(samples)
    # print(avg)
    result = [0] * N
    for i in range(N):
        result[i] = samples[i] - avg

    rounded_numbers = [round(num, 3) for num in result]
    file_name = 'TestTask_5\Remove DC component\DC_component_output.txt'
    comparesignal2.SignalSamplesAreEqual(file_name, rounded_numbers)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(indices, samples)
    plt.title('Signal before removing DC')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(indices, rounded_numbers)
    plt.title('Signal after removing DC')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def smoothing():
    input_gui = Tk()
    input_gui.geometry('350x300+820+300')
    input_gui.resizable(False, False)
    input_gui.title('Window Size')
    window_size_lbl = Label(input_gui, text='Window Size:')
    window_size_lbl.pack()
    window_size_entry = Entry(input_gui)
    window_size_entry.pack()

    def smoothed_signal():
        window_size = window_size_entry.get()
        try:
            num = int(window_size)
            indices, samples = read_file()

            smoothed_signal = [0] * (len(samples) - num + 1)
            for i in range(len(smoothed_signal)):
                smoothed_signal[i] = sum(samples[i:i + num]) / num

            print(smoothed_signal)
            print(len(smoothed_signal))
            plotting(indices[:len(smoothed_signal)], smoothed_signal, 'Smoothed Signal')

        except ValueError:
            messagebox.showwarning(title="Warning", message="Enter the Window Size")

        input_gui.destroy()

    generate_button = Button(input_gui, text='Generate Signal', command=smoothed_signal)
    generate_button.pack()
    input_gui.mainloop()


def sharpening():
    DerivativeSignal()


def delay_advance_signal():
    input_gui = Tk()
    input_gui.geometry('350x300+820+300')
    input_gui.resizable(False, False)
    input_gui.title('K Steps')
    num_of_steps_lbl = Label(input_gui, text='Number of Steps:')
    num_of_steps_lbl.pack()
    num_of_steps_entry = Entry(input_gui)
    num_of_steps_entry.pack()

    def delayAdvanced_signal():
        num_of_steps = num_of_steps_entry.get()
        try:
            num = int(num_of_steps)
            indices, samples = read_file()
            delayed_advanced_signal = [0] * len(indices)

            for i in range(len(samples)):
                delayed_advanced_signal[i] = indices[i] + num

            plt.figure(figsize=(10, 6))
            plt.plot(indices, samples, label='Original Signal')
            plt.plot(delayed_advanced_signal, samples, label=f'Delayed/Advanced Signal (k={num})')
            plt.legend()
            plt.title('Delayed/Advanced Signal')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()
        except ValueError:
            messagebox.showwarning(title="Warning", message="Enter the number of steps")

        input_gui.destroy()

    generate_button = Button(input_gui, text='Generate Signal', command=delayAdvanced_signal)
    generate_button.pack()
    input_gui.mainloop()


def fold_signal(fold_only=True):
    indices, samples = read_file()

    # folded_signal = samples[::-1]
    folded_signal = [samples[len(samples) - 1 - i] for i in range(len(samples))]
    if not fold_only:
        return indices, folded_signal
    file_name = 'TestTask_6\Shifting_and_Folding\Output_fold.txt'
    Shift_Fold_Signal(file_name, indices, folded_signal)

    plt.figure(figsize=(10, 6))
    plt.plot(indices, samples, label='Original Signal')
    plt.plot(indices, folded_signal, label='Folded Signal')
    plt.legend()
    plt.title('Folded Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def delay_advance_folded_signal():
    input_gui = Tk()
    input_gui.geometry('350x300+820+300')
    input_gui.resizable(False, False)
    input_gui.title('K Steps')
    num_of_steps_lbl = Label(input_gui, text='Number of Steps:')
    num_of_steps_lbl.pack()
    num_of_steps_entry = Entry(input_gui)
    num_of_steps_entry.pack()

    def delayAdvanced_signal():
        num_of_steps = num_of_steps_entry.get()
        try:
            num = int(num_of_steps)
            indices, folded_signal = fold_signal(fold_only=False)
            delayed_advanced_signal = [0] * len(indices)

            for i in range(len(indices)):
                delayed_advanced_signal[i] = indices[i] + num

            int_num1 = [int(num) for num in delayed_advanced_signal]
            int_num2 = [int(num) for num in folded_signal]

            # file_name = 'TestTask_6\Shifting_and_Folding\Output_ShifFoldedby500.txt'
            file_name = 'TestTask_6\Shifting_and_Folding\Output_ShiftFoldedby-500.txt'
            Shift_Fold_Signal(file_name, int_num1, int_num2)

            plt.figure(figsize=(10, 6))
            plt.plot(indices, folded_signal, label='Original Signal')
            plt.plot(delayed_advanced_signal, folded_signal, label=f'Delayed/Advanced Folded Signal (k={num})')
            plt.legend()
            plt.title('Delayed/Advanced Folded Signal')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.show()
        except ValueError:
            messagebox.showwarning(title="Warning", message="Enter the number of steps")

        input_gui.destroy()

    generate_button = Button(input_gui, text='Generate Signal', command=delayAdvanced_signal)
    generate_button.pack()
    input_gui.mainloop()


def remove_dc_td():
    indices, samples = read_file()

    dft_data = dft(samples)
    amplitude = np.abs(dft_data)
    phase = np.angle(dft_data)

    amplitude[0] = 0
    phase[0] = 0
    result = idft(amplitude, phase)

    rounded_numbers = [round(num, 3) for num in result]
    # print(f'actual output:   {rounded_numbers}')

    file_name = 'TestTask_5\Remove DC component\DC_component_output.txt'
    comparesignal2.SignalSamplesAreEqual(file_name, rounded_numbers)


def convolution():
    indices1, samples1 = read_file()
    indices2, samples2 = read_file()

    result_indices = []
    result_samples = []
    for n in range(len(indices1) + len(indices2) - 1):
        result_indices.append(indices1[0] + indices2[0] + n)

        result_sample = 0
        for k in range(len(indices1)):
            if n - k < 0 or n - k >= len(indices2):
                continue
            result_sample += samples1[k] * samples2[n - k]

        result_samples.append(result_sample)

    print(f'Result Indices: {result_indices}')
    print(f'Result Samples: {result_samples}')
    ConvTest.ConvTest(result_indices, result_samples)


def GUI():
    gui = Tk()
    gui.geometry('750x450+520+250')
    gui.resizable(False, False)
    gui.title('Main Form')
    gui.config(background='gray')

    lbl = Label(gui, text='DSP Tasks', bg='green', font=("Arial", 16), width=65)
    lbl.place(x=-20, y=10)

    btn_select_file = Button(gui, text='Select File', bg='pink', font=("Arial", 12), width=20,
                             command=signal_representation)
    btn_select_file.place(x=20, y=100)

    btn_generate_sine_signal = Button(gui, text='Generate Sine Signal', bg='lightblue', font=("Arial", 12), width=20,
                                      command=generate_sine_signal)
    btn_generate_sine_signal.place(x=20, y=150)

    btn_generate_cosine_signal = Button(gui, text='Generate Cosine Signal', bg='lightgreen', font=("Arial", 12),
                                        width=20, command=generate_cosine_signal)
    btn_generate_cosine_signal.place(x=20, y=200)

    btn_add_signal = Button(gui, text='Add Signals', bg='blue', font=("Arial", 12), width=20,
                            command=add_signals)
    btn_add_signal.place(x=300, y=100)

    btn_sub_signal = Button(gui, text='Subtact Signals', bg='green', font=("Arial", 12), width=20,
                            command=sub_signals)
    btn_sub_signal.place(x=300, y=150)

    btn_multiply_signal = Button(gui, text='Multiply Signal', bg='yellow', font=("Arial", 12), width=20,
                                 command=multiply_signal)
    btn_multiply_signal.place(x=300, y=200)

    btn_squaring_signal = Button(gui, text='Squaring Signal', bg='orange', font=("Arial", 12), width=20,
                                 command=squaring_signal)
    btn_squaring_signal.place(x=550, y=100)

    btn_Shifting_signal = Button(gui, text='Shifting Signal', bg='magenta', font=("Arial", 12), width=20,
                                 command=shift_signal)
    btn_Shifting_signal.place(x=550, y=150)

    btn_normalizes_signal = Button(gui, text='Normalizes Signal', bg='cyan', font=("Arial", 12), width=20,
                                   command=normalize_signal)
    btn_normalizes_signal.place(x=550, y=200)

    btn_accumulation_signal = Button(gui, text='Accumulation Signal', bg='purple', font=("Arial", 12), width=20,
                                     command=accumulate_signal)
    btn_accumulation_signal.place(x=420, y=250)

    btn_quantize_signal = Button(gui, text='quantization Signal', bg='cyan', font=("Arial", 12), width=20,
                                 command=quantize_signal)
    btn_quantize_signal.place(x=20, y=350)

    def apply_fourier_transform_menu():
        sampling_frequency = float(input("Enter the sampling frequency (Hz): "))
        apply_fourier_transform(sampling_frequency)

    # Add the "Frequency Domain" menu
    menu = Menu(gui)
    gui.config(menu=menu)
    frequency_domain_menu = Menu(menu)
    menu.add_cascade(label="Frequency Domain", menu=frequency_domain_menu)
    frequency_domain_menu.add_command(label="Apply Fourier Transform", command=apply_fourier_transform_menu)
    frequency_domain_menu.add_command(label="Modify Amplitude and Phase", command=modify_amplitude_phase)
    frequency_domain_menu.add_command(label="Save Frequency Components", command=save_frequency_components)
    frequency_domain_menu.add_command(label="Read and Reconstruct", command=read_and_reconstruct)
    frequency_domain_menu.add_command(label="DCT", command=dct)
    frequency_domain_menu.add_command(label="Remove DC", command=remove_dc)

    # Add the "Time Domain" menu
    time_domain_menu = Menu(menu)
    menu.add_cascade(label="Time Domain", menu=time_domain_menu)
    time_domain_menu.add_command(label="Smoothing", command=smoothing)
    time_domain_menu.add_command(label="Sharpening", command=sharpening)
    time_domain_menu.add_command(label="Delaying OR Advancing a signal", command=delay_advance_signal)
    time_domain_menu.add_command(label="Folding", command=fold_signal)
    time_domain_menu.add_command(label="Delaying OR Advancing a folded signal", command=delay_advance_folded_signal)
    time_domain_menu.add_command(label="Remove DC ", command=remove_dc_td)
    time_domain_menu.add_command(label="Convolution ", command=convolution)

    gui.mainloop()


GUI()
