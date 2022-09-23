import math

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

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


def add_label(x, y):
    for xx, yy in zip(x, y):
        if yy < 1e-8:
            continue
        if (xx + natural_f) % sampling_f < 1e-2:
            loc = 10
        else:
            loc = 4
        plt.annotate(f"{yy:.2e}", (xx, yy), textcoords="offset points", xytext=(0, loc), ha='center',
                     bbox=dict(boxstyle="square,pad=0", fc="white", ec='none'))


def compute_range(array):
    valid_array = array[array > 1E-10]
    y_actual = math.log10(np.max(valid_array))
    y_max = math.ceil(y_actual)
    if y_max < y_actual + .2:
        y_max += 1
    y_min = min(y_max - 3, math.floor(math.log10(np.min(valid_array))))
    return [10 ** y_min, 10 ** y_max]


def get_window(length: int = 512, half: bool = False, window_type: str = 'tri'):
    up_sampling_f = sampling_f * ratio
    if window_type == 'tri':
        window = signal.windows.triang(2 * ratio - 1)
    else:
        bin_num = 32 * ratio
        if window_type == 'flattop':
            window = signal.windows.flattop(bin_num + 1)
        elif window_type == 'blackmanharris':
            window = signal.windows.blackmanharris(bin_num + 1)
        elif window_type == 'hann':
            window = signal.windows.hann(bin_num + 1)
        elif window_type == 'hamming':
            window = signal.windows.hamming(bin_num + 1)
        elif window_type == 'kaiser':
            window = signal.windows.kaiser(bin_num + 1, 9)
        elif window_type == 'cheb':
            window = signal.windows.chebwin(bin_num + 1, 80)
        else:
            raise ValueError(f"Unknown window type: {window_type}")

        cutoff = 1 / ratio
        window *= np.sinc(cutoff * (np.linspace(0, bin_num, bin_num + 1) - bin_num // 2))
        window /= np.sum(window)
        window *= ratio

    if half:
        window_amp = np.fft.rfft(window, 2 * length)
        window_freq = np.fft.rfftfreq(2 * len(window_amp) - 2, 1 / up_sampling_f)
    else:
        window_amp = np.fft.fftshift(np.fft.fft(window, length))
        window_freq = np.fft.fftshift(np.fft.fftfreq(len(window_amp), 1 / up_sampling_f))

    window_amp /= np.max(np.abs(window_amp))

    return window, window_freq, window_amp


def get_waveform(samples: int):
    x = np.linspace(0, duration, samples, endpoint=False)
    y = np.sin(2 * np.pi * natural_f * x)
    return x, y


def zero_stuff(x, y, scale):
    up_x = np.linspace(0, duration, scale * len(x), endpoint=False)
    up_y = np.zeros(len(up_x))
    up_y[::scale] = y
    return up_x, up_y


def perform_fft(y, scale=1):
    amplitude = 2 * np.fft.rfft(y) / len(y) * scale
    freq = np.fft.rfftfreq(2 * len(amplitude) - 2, 1 / sampling_f / scale)
    return freq, amplitude


def perform_computation():
    o_time, o_sine_wave = get_waveform(int(duration * sampling_f))
    o_freq, o_amplitude = perform_fft(o_sine_wave)

    up_time, up_sine_wave = zero_stuff(o_time, o_sine_wave, ratio)
    up_freq, up_amplitude = perform_fft(up_sine_wave, ratio)

    tri_window, window_freq, window_amp = get_window()
    window_amp = np.abs(window_amp)

    conv = np.convolve(up_sine_wave, tri_window, mode='same')
    conv_fft = np.abs(2 * np.fft.rfft(conv) / len(conv))
    conv_v = np.multiply(np.abs(up_freq), conv_fft)
    conv_a = np.multiply(np.abs(up_freq), conv_v)

    up_sampling_f = sampling_f * ratio
    margin = duration / 30
    xlim_time = [-margin, duration + margin]
    margin *= up_sampling_f / duration / 2
    xlim_freq = [-margin, up_sampling_f / 2 + margin]
    ylim_time = [-1.1, 1.1]

    fig = plt.figure(figsize=(6, 3), dpi=200)

    fig.add_subplot(211)
    add_stem(rf'original sine wave $p[n]$ with $f={natural_f}$ Hz and $f_s={sampling_f}$ Hz', o_time, o_sine_wave,
             'Time (s)', 'Amplitude')
    plt.xlim(xlim_time)
    plt.ylim(ylim_time)

    fig.add_subplot(212)
    add_stem('original spectrum', o_freq, np.abs(o_amplitude), 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)

    fig.tight_layout()
    fig.savefig('../PIC/PureSineOrigin.pdf', format='pdf')
    fig = plt.figure(figsize=(6, 3), dpi=200)

    fig.add_subplot(211)
    add_stem(rf'extended sine wave $p_e[n]$ with $L={ratio}$', up_time, up_sine_wave, 'Time (s)', 'Amplitude')
    plt.xlim(xlim_time)
    plt.ylim(ylim_time)

    fig.add_subplot(212)
    add_stem(rf'extended spectrum $L={ratio}$', up_freq, np.abs(up_amplitude), 'Frequency (Hz)', 'Amplitude')
    plt.xlim(xlim_freq)

    fig.tight_layout()
    fig.savefig('../PIC/PureSineExtended.pdf', format='pdf')
    fig = plt.figure(figsize=(6, 1.5), dpi=200)

    fig.add_subplot(111)
    add_plot(rf'triangular window', window_freq, 20 * np.log10(np.maximum(window_amp, 1e-12)), 'Frequency (Hz)',
             'Amplitude (dB)')
    plt.plot(window_freq, 20 * np.log10(np.maximum(np.square(np.sinc(window_freq / sampling_f)), 1e-12)),
             linestyle='dashed')
    plt.legend(['DFT', 'analytical'])
    plt.xlim([-xlim_freq[1], xlim_freq[1]])
    plt.ylim([-60, 10])

    fig.tight_layout()
    fig.savefig('../PIC/TriangularWindow.pdf', format='pdf')

    fig = plt.figure(figsize=(6, 3), dpi=200)

    fig.add_subplot(211)
    add_stem(rf'convolution/interpolation $p_i[n]$', up_time, conv, 'Time (s)', 'Amplitude', True)
    plt.xlim(xlim_time)
    plt.ylim(ylim_time)

    fig.add_subplot(212)
    add_stem(rf'convolution/interpolation spectrum $L={ratio}$', up_freq, conv_fft, 'Frequency (Hz)', 'Amplitude')
    add_label(up_freq, conv_fft)
    plt.xlim(xlim_freq)
    plt.yscale('log')
    plt.ylim(compute_range(conv_fft))

    fig.tight_layout()
    fig.savefig('../PIC/Convolution.pdf', format='pdf')

    np.savetxt('../PIC/Convolution.csv', conv_fft[conv_fft > 1e-3], delimiter=',', header='Amplitude')
    np.savetxt('../PIC/ConvolutionW.csv', conv_v[conv_v > 1e-3], delimiter=',', header='Amplitude')
    np.savetxt('../PIC/ConvolutionWW.csv', conv_a[conv_a > 1e-3], delimiter=',', header='Amplitude')

    window_amp = np.fft.rfft(tri_window, len(up_sine_wave))
    window_amp /= np.max(window_amp)
    product = np.abs(up_amplitude * window_amp)
    np.savetxt('../PIC/TriMultiply.csv', product[product > 1e-3], delimiter=',', header='Amplitude')


if __name__ == '__main__':
    perform_computation()
