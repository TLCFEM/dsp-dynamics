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
    up_sampling_f = sampling_f * ratio

    o_time = np.linspace(0, duration, samples, endpoint=False)
    o_sine_wave = np.sin(2 * np.pi * natural_f * o_time)

    up_time = np.linspace(0, duration, ratio * samples, endpoint=False)
    up_sine_wave = np.zeros(ratio * samples)
    up_sine_wave[::ratio] = o_sine_wave

    o_amplitude = 2 * np.fft.rfft(o_sine_wave) / len(o_sine_wave)
    o_freq = np.fft.rfftfreq(2 * len(o_amplitude) - 2, 1 / sampling_f)

    up_amplitude = 2 * np.fft.rfft(up_sine_wave) / len(up_sine_wave) * ratio
    up_freq = np.fft.rfftfreq(2 * len(up_amplitude) - 2, 1 / up_sampling_f)

    tri_window = signal.windows.triang(2 * ratio - 1)
    window_amp = np.abs(np.fft.fftshift(np.fft.fft(tri_window, 512)))
    window_amp /= np.max(window_amp)
    window_freq = np.fft.fftshift(np.fft.fftfreq(len(window_amp), 1 / up_sampling_f))

    conv = np.convolve(up_sine_wave, tri_window, mode='same')
    conv_fft = np.abs(2 * np.fft.rfft(conv) / len(conv))
    conv_v = np.multiply(np.abs(up_freq), conv_fft)
    conv_a = np.multiply(np.abs(up_freq), conv_v)

    margin = duration / 30
    xlim_time = [-margin, duration + margin]
    margin *= up_sampling_f / duration / 2
    xlim_freq = [-margin, up_sampling_f / 2 + margin]
    ylim_time = [-1.1, 1.1]

    fig = plt.figure(figsize=(6, 3), dpi=200)

    fig.add_subplot(211)
    add_stem(rf'original sine wave $u[n]$ with $f={natural_f}$ Hz and $f_s={sampling_f}$ Hz', o_time, o_sine_wave,
             'Time (s)', 'Amplitude')
    plt.xlim(xlim_time)
    plt.ylim(ylim_time)

    fig.add_subplot(212)
    add_stem('original spectrum', o_freq, np.abs(o_amplitude), 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)

    fig.tight_layout()
    fig.savefig('../PIC/PureSineOrigin.eps', format='eps')
    fig = plt.figure(figsize=(6, 3), dpi=200)

    fig.add_subplot(211)
    add_stem(rf'extended sine wave $u_e[n]$ with $L={ratio}$', up_time, up_sine_wave, 'Time (s)', 'Amplitude')
    plt.xlim(xlim_time)
    plt.ylim(ylim_time)

    fig.add_subplot(212)
    add_stem(rf'extended spectrum $L={ratio}$', up_freq, np.abs(up_amplitude), 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)

    fig.tight_layout()
    fig.savefig('../PIC/PureSineExtended.eps', format='eps')
    fig = plt.figure(figsize=(6, 1.5), dpi=200)

    fig.add_subplot(111)
    add_plot(rf'triangular window', window_freq, 20 * np.log10(np.maximum(window_amp, 1e-12)), 'Frequency (Hz)',
             'Amplitude (dB)')
    plt.xlim([-xlim_freq[1], xlim_freq[1]])
    plt.ylim([-60, 10])

    fig.tight_layout()
    fig.savefig('../PIC/TriangularWindow.eps', format='eps')

    fig = plt.figure(figsize=(6, 6), dpi=200)

    fig.add_subplot(411)
    add_stem(rf'convolution/interpolation $u_i[n]$', up_time, conv, 'Time (s)', 'Amplitude', True)
    plt.xlim(xlim_time)
    plt.ylim(ylim_time)

    fig.add_subplot(412)
    add_stem(rf'convolution/interpolation spectrum $L={ratio}$', up_freq, conv_fft, 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)
    plt.yscale('log')
    plt.ylim(compute_range(conv_fft))

    fig.add_subplot(413)
    add_stem(rf'$\omega{{}}u_i[n]$ spectrum $L={ratio}$', up_freq, conv_v, 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)
    plt.yscale('log')
    plt.ylim(compute_range(conv_v))

    fig.add_subplot(414)
    add_stem(rf'$\omega^2{{}}u_i[n]$ spectrum $L={ratio}$', up_freq, conv_a, 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)
    plt.yscale('log')
    plt.ylim(compute_range(conv_a))

    fig.tight_layout()
    fig.savefig('../PIC/Convolution.eps', format='eps')

    np.savetxt('../PIC/Convolution.csv', conv_fft[conv_fft > 1e-3], delimiter=',', header='Amplitude')
    np.savetxt('../PIC/ConvolutionW.csv', conv_v[conv_v > 1e-3], delimiter=',', header='Amplitude')
    np.savetxt('../PIC/ConvolutionWW.csv', conv_a[conv_a > 1e-3], delimiter=',', header='Amplitude')

    window_amp = np.fft.rfft(tri_window, len(up_sine_wave))
    window_amp /= np.max(window_amp)
    product = np.abs(up_amplitude * window_amp)
    np.savetxt('../PIC/TriMultiply.csv', product[product > 1e-3], delimiter=',', header='Amplitude')


if __name__ == '__main__':
    perform_computation()
