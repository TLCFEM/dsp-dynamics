import math
from itertools import cycle

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import PureSine

matplotlib.rcParams.update({'font.size': 6})

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


def constant_kernel(f, f_n, a):
    beta = f / f_n
    zeta = a
    factor = 2 * zeta * beta
    numerator = factor * 1j
    denom = 1 - beta ** 2 + numerator
    return numerator / denom


def mass_kernel(f, f_n, a):
    omega = 2 * math.pi * f
    omega_n = 2 * math.pi * f_n
    factor = 2 * a * omega
    numerator = factor * 1j
    denom = omega_n ** 2 - omega ** 2 + numerator
    return numerator / denom


def stiffness_kernel(f, f_n, a):
    beta = f / f_n
    factor = 2 * a * 2 * math.pi * f
    numerator = factor * 1j
    denom = 1 - beta ** 2 + numerator
    return numerator / denom


def inertial_kernel(f, f_n, a):
    beta = f / f_n
    zeta = a
    numerator = -beta ** 2
    denom = 1 - beta ** 2 + 2 * beta * zeta * 1j
    return numerator / denom


LS = get_line_style()


def compute_response(damping_type, a, freq_n: float):
    freq = np.linspace(0, max_frequency // 2, max_frequency // 2 + 1)

    if damping_type == 'Constant':
        magnitude = constant_kernel(freq, freq_n, a)
    elif damping_type == 'Mass':
        magnitude = mass_kernel(freq, freq_n, a)
    elif damping_type == 'Stiffness':
        magnitude = stiffness_kernel(freq, freq_n, a)
    elif damping_type == 'Inertial':
        magnitude = inertial_kernel(freq, freq_n, a)
    elif damping_type == 'InertialStiffness':
        magnitude = inertial_kernel(freq, freq_n, a * 2 * math.pi * freq_n)
    elif damping_type == 'InertialMass':
        magnitude = inertial_kernel(freq, freq_n, a / 2 / math.pi / freq_n)
    else:
        raise ValueError('Unknown damping type')

    return freq, magnitude


def perform_analysis(damping_type: str = 'Stiffness', a: float = .001):
    if damping_type == 'Constant':
        fig = plt.figure(figsize=(6, 1.8), dpi=200)
    else:
        fig = plt.figure(figsize=(6, 1.8), dpi=200)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Magnitude $|\hat{k_v}|$')
    plt.grid(True, which='both', linewidth=.2, linestyle='-')
    plt.gca().xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

    all_freq = np.arange(0, 1000, 100)
    all_freq = all_freq[1:-1]
    all_freq = [f + 100 for f in all_freq]
    for f in all_freq:
        freq, magnitude = compute_response(damping_type, a, f)
        plt.plot(
            freq,
            np.abs(magnitude),
            linestyle=next(LS),
            linewidth=1.4,
        )

    legend_location = 'lower right'
    n_col = 2

    if damping_type == 'Stiffness':
        plt.title(rf'frequency response of {damping_type.lower()} proportional damping ($a_1={a}$)')
        plt.legend(
            [rf'$f_n={v:3.1f}$ Hz, $\omega_n={v * 2 * np.pi:06.1f}$ rad/s, $\zeta={a * v * 2 * np.pi:.3f}$' for v in
             all_freq], handlelength=3, ncol=n_col, loc=legend_location)
        plt.legend(
            [rf'$f_n={v:3.1f}$ Hz, $\zeta={a * v * 2 * np.pi:.3f}$' for v in
             all_freq], handlelength=3, ncol=n_col, loc=legend_location)
    elif damping_type == 'Mass':
        plt.title(rf'frequency response of {damping_type.lower()} proportional damping ($a_0={a}$)')
        plt.legend(
            [rf'$f_n={v:3.1f}$ Hz, $\zeta={a / (v * 2 * np.pi):2.1e}$' for v in
             all_freq], handlelength=3, ncol=n_col, loc=legend_location)
    else:
        plt.title(rf'frequency response of constant damping ($\zeta={a}$)')
        plt.legend([rf'$f_n={v}$ Hz' for v in all_freq], handlelength=3, ncol=n_col, loc=legend_location)

    plt.xlim([0, max_frequency // 2])
    fig.tight_layout()
    fig.savefig(f'../PIC/{damping_type}Proportional{int(1e5 * a)}.pdf', format='pdf')


if __name__ == '__main__':
    perform_analysis('Constant', .0002)
    perform_analysis('Constant', .002)
    perform_analysis('Constant', .02)
    perform_analysis('Constant', .2)
    perform_analysis('Stiffness', .0001)
    perform_analysis('Mass', 5)
