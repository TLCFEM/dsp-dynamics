from itertools import cycle

import h5py as h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from FundamentalSolution import compute_response, get_line_style
from PureSine import compute_range, duration, get_waveform, get_window, natural_f, ratio, sampling_f, zero_stuff

matplotlib.rcParams.update({'font.size': 6})

__LOG_SCALE__ = True


def get_list():
    freq_list = []
    for i in range(6):
        freq = 200 * i + 25
        if 0 < freq < 1000:
            freq_list.append(freq)
        freq = 200 * i - 25
        if 0 < freq < 1000:
            freq_list.append(freq)
    return freq_list


def get_loc():
    for i in cycle([10, 14]):
        yield i


__LOC__ = get_loc()

LS = get_line_style()

__SAVE__ = True


def plot(damping_type, a, freq_n: float, win_type: str = 'tri'):
    _, freq, window_amp = get_window(ratio * sampling_f // 2, True, win_type)
    mask = np.isin(freq, get_list())

    _, amp = compute_response(damping_type, a, freq_n)

    total_amp = np.abs(amp * window_amp)

    fig = plt.figure(figsize=(6, 3), dpi=200)
    plt.title(rf'{damping_type.lower()} proportional damping with {win_type} window, $a_1={a}$, $f_n={freq_n}$ Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Magnitude $|\hat{m}|$')
    plt.plot(freq, np.maximum(1e-12, total_amp))
    cherry = total_amp[mask]
    plt.scatter(freq[mask], cherry)
    for xx, yy in zip(freq[mask], cherry):
        plt.annotate(f"{yy:.2e}", (xx, yy), textcoords="offset points", xytext=(0, next(__LOC__)), ha='center',
                     bbox=dict(boxstyle="square,pad=0", fc="white", ec='none'))
    print(np.abs(amp[mask]))
    if __LOG_SCALE__:
        plt.yscale('log')
        plt.ylim(compute_range(cherry))
    plt.grid(True)
    fig.tight_layout()
    if __SAVE__:
        fig.savefig(f'../PIC/{damping_type}DampingForce{int(freq_n)}-{int(1e5 * a)}.eps', format='eps')
    else:
        plt.show()

    return freq[mask], cherry


def surface(damping_type, a, win_type: str = 'tri'):
    _, freq, window_amp = get_window(ratio * sampling_f // 2, True, win_type)
    x, y = np.meshgrid(freq, freq)

    array = np.zeros((len(freq), len(freq)))
    for i in range(len(window_amp)):
        _, amp = compute_response(damping_type, a, max(float(i), .01))
        array[:, i] = np.abs(amp * window_amp)

    fig, ax = plt.subplots()
    surf = ax.pcolormesh(x, y, np.log10(np.maximum(1e-14, array)).T, cmap='PiYG')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Natural Frequency (Hz)')
    fig.colorbar(surf, aspect=40)
    fig.tight_layout()
    fig.show()


def signal(t):
    o_time = np.linspace(0, t, int(t * sampling_f), endpoint=False)
    o_sine_wave = np.sin(2 * np.pi * natural_f * o_time)

    return np.vstack((o_time, o_sine_wave)).T


def process_result(x, y):
    fig = plt.figure(figsize=(6, 4), dpi=200)
    fig.add_subplot(211)
    plt.title(rf'damping force history of the SDOF system')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Damping Force $F_d$')
    with h5py.File('../MODEL/PureSine/Interpolation.h5', 'r') as f:
        data = f['R2-DF']['R2-DF2']
        plt.plot(data[:, 0], data[:, 1], linewidth=1)
        plt.xlim([0, np.max(data[:, 0])])

    with h5py.File('../MODEL/PureSine/Analytical.h5', 'r') as f:
        data = f['R2-DF']['R2-DF2']
        plt.plot(data[:, 0], data[:, 1], linewidth=2, linestyle='--', c='#e41a1c')

    plt.grid(True, which='both')
    plt.legend(['interpolated external load', 'analytical external load'])

    fig.add_subplot(212)
    plt.title('frequency response of damping force of the SDOF system')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Magnitude of Damping Force $F_d$')
    plt.scatter(x, y, facecolors='none', edgecolors='grey')

    with h5py.File('../MODEL/PureSine/Interpolation.h5', 'r') as f:
        data = f['R2-DF']['R2-DF2']
        o_amplitude = 2 * np.fft.rfft(data[:, 1]) / len(data[:, 1])
        o_freq = np.fft.rfftfreq(2 * len(o_amplitude) - 2, data[1, 0] - data[0, 0])

        plt.plot(o_freq, np.abs(o_amplitude))

        plt.xlim([0, np.max(o_freq)])

    with h5py.File('../MODEL/PureSine/Analytical.h5', 'r') as f:
        data = f['R2-DF']['R2-DF2']
        o_amplitude = 2 * np.fft.rfft(data[:, 1]) / len(data[:, 1])
        o_freq = np.fft.rfftfreq(2 * len(o_amplitude) - 2, data[1, 0] - data[0, 0])

        plt.plot(o_freq, np.abs(o_amplitude), linewidth=2, linestyle='--', c='#e41a1c')

    plt.legend(['theoretical values (for interpolated load)', 'interpolated external load', 'analytical external load'])
    plt.grid(True, which='both')
    fig.tight_layout()

    if __SAVE__:
        fig.savefig(f'../PIC/InterpolationExample.eps', format='eps')
    else:
        plt.show()


def plot_window(win_type: str = 'tri'):
    window, window_freq, window_amp = get_window(500, False, win_type)
    window_amp = np.abs(window_amp)
    fig = plt.figure(figsize=(6, 3), dpi=200)
    fig.add_subplot(211)
    plt.title(f'{win_type} window function')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.plot(window_freq, 20 * np.log10(np.maximum(window_amp, 1e-12)))
    plt.grid(True)
    plt.xlim([-1000, 1000])

    fig.add_subplot(212)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    o_time, o_sine_wave = get_waveform(int(duration * sampling_f))
    up_time, up_sine_wave = zero_stuff(o_time, o_sine_wave, ratio)

    markerline, stemline, baseline = plt.stem(up_time, up_sine_wave, markerfmt='ro', linefmt='--')
    plt.setp(stemline, linewidth=0)
    plt.setp(markerline, markersize=2)
    plt.setp(stemline, color='#377eb8')
    plt.setp(baseline, color='#555555')
    plt.setp(baseline, linewidth=.1)
    plt.grid(True)

    plt.plot(up_time, np.convolve(up_sine_wave, window, mode='same'))

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    plot('Stiffness', .0001, 200, 'blackmanharris')
    plot('Stiffness', .0001, 200, 'hann')
    plot('Stiffness', .0001, 200, 'hamming')
    plot('Stiffness', .0001, 200, 'kaiser')
    plot('Mass', .001, 225, 'cheb')
    xx, yy = plot('Stiffness', .0001, 200)

    np.savetxt('../MODEL/PureSine/motion', signal(5), fmt='%.15e')

    process_result(xx, yy)

    plot_window('tri')

    surface('Mass', .001, 'tri')
