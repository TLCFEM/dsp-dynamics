import math

import matplotlib
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 6})

natural_f = 25
sampling_f = 200
duration = .2
ratio = 10


def add_stem(title, x, y, x_label, y_label, hide_stem=False):
    markerline, stemline, baseline = plt.stem(x, y, markerfmt='ro', linefmt='--')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.setp(stemline, linewidth=2e-2 if hide_stem else 1)
    plt.setp(stemline, color='#e41a1c')
    plt.setp(markerline, markersize=2)
    plt.setp(markerline, color='#377eb8')
    plt.setp(baseline, color='#555555')
    plt.setp(baseline, linewidth=1)
    plt.grid(True)


def add_plot(title, x, y, x_label, y_label):
    baseline = plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.setp(baseline, linewidth=1)
    plt.setp(baseline, color='#e41a1c')
    plt.grid(True)


def compute_range(array):
    valid_array = array[array > 1E-8]
    y_actual = math.log10(np.max(valid_array))
    y_max = math.ceil(y_actual)
    if y_max < y_actual + .2:
        y_max += 1
    y_min = min(y_max - 3, math.floor(math.log10(np.min(valid_array))))
    return [10 ** y_min, 10 ** y_max]


def perform_computation():
    samples = int(duration * sampling_f)

    time = np.linspace(0, duration, samples, endpoint=False)
    amplitude = np.sin(2 * np.pi * natural_f * time)

    up_time = np.linspace(0, duration, ratio * samples, endpoint=False)
    up_amplitude = np.zeros(ratio * samples)
    up_amplitude[::ratio] = amplitude

    original = 2 * np.fft.rfft(amplitude) / samples
    freq = np.fft.rfftfreq(samples, 1 / sampling_f)

    up_original = 2 * np.fft.rfft(up_amplitude) / samples
    up_freq = np.fft.rfftfreq(ratio * samples, 1 / sampling_f / ratio)

    tri_window = signal.windows.triang(2 * ratio - 1)
    print(tri_window)
    window_amp = np.abs(np.fft.fftshift(np.fft.fft(tri_window, 1024) / len(tri_window)))
    window_amp /= np.max(window_amp)
    window_freq = np.linspace(-ratio * sampling_f / 2, ratio * sampling_f / 2, len(window_amp))

    conv = np.convolve(up_amplitude, tri_window, mode='same')
    conv_fft = np.abs(2 * np.fft.rfft(conv) / samples / ratio)
    conv_v = np.multiply(np.abs(up_freq), conv_fft)
    conv_a = np.multiply(np.abs(up_freq), conv_v)

    margin = duration / 30
    xlim_time = [-margin, duration + margin]
    margin *= ratio * sampling_f / duration / 2
    xlim_freq = [-margin, ratio * sampling_f / 2 + margin]
    ylim_time = [-1.1, 1.1]

    fig = plt.figure(figsize=(6, 3), dpi=200)

    ax1 = fig.add_subplot(211)
    add_stem(rf'original sine wave $u[n]$ with $f={natural_f}$ Hz and $f_s={sampling_f}$ Hz', time, amplitude, 'Time (s)',
             'Amplitude')
    plt.xlim(xlim_time)
    plt.ylim(ylim_time)

    ax3 = fig.add_subplot(212)
    add_stem('original spectrum', freq, np.abs(original), 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)

    fig.tight_layout()
    fig.savefig('../PIC/PureSineOrigin.eps', format='eps')
    fig = plt.figure(figsize=(6, 3), dpi=200)

    ax2 = fig.add_subplot(211)
    add_stem(rf'extended sine wave $u_e[n]$ with $L={ratio}$', up_time, up_amplitude, 'Time (s)', 'Amplitude')
    plt.xlim(xlim_time)
    plt.ylim(ylim_time)

    ax4 = fig.add_subplot(212)
    add_stem(rf'extended spectrum $L={ratio}$', up_freq, np.abs(up_original), 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)

    fig.tight_layout()
    fig.savefig('../PIC/PureSineExtended.eps', format='eps')
    fig = plt.figure(figsize=(6, 1.5), dpi=200)

    ax1 = fig.add_subplot(111)
    add_plot(rf'triangular window', window_freq, 20 * np.log10(np.maximum(window_amp, 1e-12)), 'Frequency (Hz)',
             'Amplitude (dB)')
    plt.xlim([-xlim_freq[1], xlim_freq[1]])
    plt.ylim([-60, 10])

    fig.tight_layout()
    fig.savefig('../PIC/TriangularWindow.eps', format='eps')

    fig = plt.figure(figsize=(6, 6), dpi=200)

    ax1 = fig.add_subplot(411)
    add_stem(rf'convolution/interpolation $u_i[n]$', up_time, conv, 'Time (s)', 'Amplitude', True)
    plt.xlim(xlim_time)
    plt.ylim(ylim_time)

    ax2 = fig.add_subplot(412)
    add_stem(rf'convolution/interpolation spectrum $L={ratio}$', up_freq, conv_fft, 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)
    plt.yscale('log')
    plt.ylim(compute_range(conv_fft))

    np.savetxt('../PIC/Convolution.csv', np.vstack((up_freq, conv_fft)).T, delimiter=',', header='Frequency,Amplitude')

    ax3 = fig.add_subplot(413)
    add_stem(rf'$\omega{{}}u_i[n]$ spectrum $L={ratio}$', up_freq,
             conv_v, 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)
    plt.yscale('log')
    plt.ylim(compute_range(conv_v))

    ax3 = fig.add_subplot(414)
    add_stem(rf'$\omega^2{{}}u_i[n]$ spectrum $L={ratio}$', up_freq,
             conv_a, 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)
    plt.yscale('log')
    plt.ylim(compute_range(conv_a))

    fig.tight_layout()
    fig.savefig('../PIC/Convolution.eps', format='eps')


if __name__ == '__main__':
    perform_computation()
