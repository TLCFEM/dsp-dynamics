from itertools import cycle

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from DFT import PureSine

matplotlib.rcParams.update({'font.size': 6})


def get_kernel(omega_n, zeta):
    omega_d = omega_n * np.sqrt(1 - zeta ** 2)

    def kernel(t):
        return np.exp(-zeta * omega_n * t) * np.sin(omega_d * t) / omega_d

    return kernel


def stiffness_proportional(omega_n, a):
    return get_kernel(omega_n, a * omega_n)


def mass_proportional(omega_n, a):
    return get_kernel(omega_n, a / omega_n)


def get_amplitude(damping_type, omega_n, a, time):
    if damping_type == 'Stiffness':
        amplitude = stiffness_proportional(omega_n, a)(time)
    elif damping_type == 'Mass':
        amplitude = mass_proportional(omega_n, a)(time)
    else:
        amplitude = mass_proportional(omega_n, a * omega_n)(time)

    amplitude = np.abs(np.fft.rfft(amplitude))

    return amplitude


multiplier = 3
duration = 20


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


LS = get_line_style()


def perform_analysis(damping_type: str = 'Stiffness', a: float = .001):
    if damping_type == 'Constant':
        fig = plt.figure(figsize=(6, 2), dpi=200)
        plt.title(rf'Constant Damping Ratio ($\zeta={a}$)')
    else:
        fig = plt.figure(figsize=(6, 3), dpi=200)
        plt.title(rf'{damping_type} Proportional ($a={a}$, $m=1$)')
    plt.xlabel('Normalised Frequency')
    plt.ylabel('Amplitude')
    # plt.yscale('log')
    plt.grid(True, which='major', linewidth=.2, linestyle='-')

    def compute_response(freq):
        omega_n = 2 * np.pi * freq
        sampling_f = max(2 * multiplier * freq, PureSine.sampling_f * PureSine.ratio)
        samples = int(duration * sampling_f)

        time = np.linspace(0, duration, samples, endpoint=False)
        amplitude = get_amplitude(damping_type, omega_n, a, time)

        freq_range = np.fft.rfftfreq(samples, 1 / sampling_f)
        x_range = freq_range / freq

        plt.plot(
            x_range,
            amplitude * omega_n * 2 * a * freq,
            linestyle=next(LS),
            linewidth=1.4,
        )
        plt.xlim([0, multiplier])

    if damping_type == 'Stiffness':
        all_freq = np.arange(0, min(1000, 1 / a / 2 / np.pi), 100)
    elif damping_type == 'Mass':
        all_freq = np.linspace(0, 50, 10, endpoint=False)
    else:
        all_freq = np.linspace(0, 200, 3, endpoint=False)

    for f in all_freq[1:-1]:
        compute_response(f)

    if damping_type == 'Stiffness':
        plt.legend(
            [rf'$f_n={v:.1f}$ Hz, $\omega_n={v * 2 * np.pi:.1f}$ Hz, $\zeta={a * v * 2 * np.pi:.3f}$' for v in
             all_freq[1:-1]], handlelength=3, ncol=1, loc='upper right')
        plt.ylabel('Velocity Amplitude')
    elif damping_type == 'Mass':
        plt.legend([rf'$\omega_n={v:.1f}$ Hz, $\zeta={a / v:.1e}$' for v in all_freq[1:-1]], handlelength=3, ncol=2,
                   loc='upper right')
    else:
        plt.legend([rf'$\omega_n$'], handlelength=3, ncol=2, loc='upper right')
        plt.xlabel('Natural Frequency')

    fig.tight_layout()
    fig.show()
    fig.savefig(f'../PIC/{damping_type}Proportional{int(1e5 * a)}.eps', format='eps')


if __name__ == '__main__':
    perform_analysis('Constant', .02)
    perform_analysis('Stiffness', .0001)
