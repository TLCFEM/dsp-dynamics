import math
from itertools import cycle

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import PureSine

matplotlib.rcParams.update({'font.size': 6})

__NORMALISED__: bool = True


def get_kernel(omega_n, zeta):
    def kernel(t):
        omega_d = omega_n * np.sqrt(1. - zeta ** 2)
        return np.exp(-zeta * omega_n * t) * np.sin(omega_d * t) / omega_d

    return kernel


def stiffness_proportional(omega_n, a):
    return get_kernel(omega_n, a * omega_n)


def mass_proportional(omega_n, a):
    return get_kernel(omega_n, a / omega_n)


def get_amplitude(damping_type, omega_n, a, time):
    if damping_type == 'Stiffness':
        history = stiffness_proportional(omega_n, a)(time)
    elif damping_type == 'Mass':
        history = mass_proportional(omega_n, a)(time)
    else:
        history = get_kernel(omega_n, a)(time)

    amplitude = 2 * np.abs(np.fft.rfft(history, 2 ** (math.ceil(math.log2(len(history))) + 1))) / len(time)

    return amplitude


max_frequency = PureSine.sampling_f * PureSine.ratio


def get_line_style():
    ls_tuple = [
        ('solid', (0, ())),
        ('loosely dotted', (0, (1, 4))),
        ('dotted', (0, (1, 2))),
        ('densely dotted', (0, (1, 1))),

        ('loosely dashed', (0, (5, 4))),
        ('dashed', (0, (5, 2))),
        ('densely dashed', (0, (5, 1))),

        ('loosely dashdotted', (0, (3, 4, 1, 4))),
        ('dashdotted', (0, (3, 2, 1, 2))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        ('loosely dashdotdotted', (0, (3, 4, 1, 4, 1, 4))),
        ('dashdotdotted', (0, (3, 2, 1, 2, 1, 2))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
    ]

    for v in cycle(ls_tuple):
        yield v[1]


def compute_magnitude(freq_n, zeta, freq):
    omega = 2 * np.pi * freq
    omega_n = 2 * np.pi * freq_n
    ratio = omega / omega_n
    return 1 / np.sqrt((1 - ratio ** 2) ** 2 + (2 * zeta * ratio) ** 2)


LS = get_line_style()


def perform_analysis(damping_type: str = 'Stiffness', a: float = .001):
    if damping_type == 'Constant':
        fig = plt.figure(figsize=(6, 2), dpi=200)
    else:
        fig = plt.figure(figsize=(6, 3), dpi=200)
    plt.xlabel('Frequency (Hz)')
    if __NORMALISED__:
        plt.ylabel(r'Normalised Magnitude $u/u_{st}$')
    else:
        plt.ylabel(r'Magnitude $u$')
    plt.yscale('log')
    plt.grid(True, which='both', linewidth=.2, linestyle='-')
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

    def compute_response(freq):
        omega_n = 2 * np.pi * freq
        sampling_f = max_frequency  # Hz

        duration = -np.log2(1e-6) / a

        if damping_type == 'Constant':
            duration /= omega_n
        elif damping_type == 'Stiffness':
            duration /= omega_n * omega_n

        samples = int(duration * sampling_f)

        time = np.linspace(0, duration, samples, endpoint=False)
        amplitude = get_amplitude(damping_type, omega_n, a, time)

        freq_range = np.fft.rfftfreq(2 * len(amplitude) - 2, 1 / sampling_f)  # Hz

        if __NORMALISED__:
            amplitude /= amplitude[0]

        plt.plot(
            freq_range,
            amplitude,
            linestyle=next(LS),
            linewidth=1.4,
        )

        # if damping_type == 'Constant':
        plt.plot(
            freq_range,
            compute_magnitude(freq, omega_n * a, freq_range),
            linestyle=next(LS),
            linewidth=1,
        )

    if damping_type == 'Stiffness':
        all_freq = np.arange(0, min(100, 1 / a / 2 / np.pi), 10)
    elif damping_type == 'Mass':
        all_freq = np.arange(0, 1000, 100)
    else:
        all_freq = [0, 200, 0]

    all_freq = all_freq[1:-1]
    for f in all_freq:
        compute_response(f)

    if __NORMALISED__:
        legend_location = 'lower left'
        n_col = 2
    else:
        legend_location = 'lower left'
        n_col = 2

    if damping_type == 'Stiffness':
        plt.title(rf'frequency response of {damping_type.lower()} proportional damping ($a_1={a}$, $m=1$)')
        plt.legend(
            [rf'$f_n={v:3.1f}$ Hz, $\omega_n={v * 2 * np.pi:06.1f}$ rad/s, $\zeta={a * v * 2 * np.pi:.3f}$' for v in
             all_freq], handlelength=3, ncol=n_col, loc=legend_location)
    elif damping_type == 'Mass':
        plt.title(rf'frequency response of {damping_type.lower()} proportional damping ($a_0={a}$, $m=1$)')
        plt.legend(
            [rf'$f_n={v:3.1f}$ Hz, $\zeta={a / (v * 2 * np.pi):2.1e}$' for v in
             all_freq], handlelength=3, ncol=2, loc='lower left')
    else:
        plt.title(rf'frequency response of constant damping ($\zeta={a}$)')
        plt.legend([rf'$f_n={all_freq[0]}$ Hz'], handlelength=3, loc='upper right')

    plt.xlim([0, max_frequency / 2])
    fig.tight_layout()
    # fig.show()
    fig.savefig(f'../PIC/{damping_type}Proportional{int(1e5 * a)}.eps', format='eps')


if __name__ == '__main__':
    max_frequency = 1000
    perform_analysis('Constant', .01)
    max_frequency = 2000
    perform_analysis('Stiffness', .00001)
    perform_analysis('Mass', 2)
