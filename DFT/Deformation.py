import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from FundamentalSolution import get_line_style

matplotlib.rcParams.update({'font.size': 6})

LS = get_line_style()

__SAVE__ = True


def z_transform(gamma):
    beta = .25 * (.5 + gamma) ** 2
    a = np.array([1, -2, 1])
    b = np.array([beta, .5 + gamma - 2 * beta, .5 + beta - gamma])
    w, h = signal.freqz(b, a, worN=np.linspace(0, np.pi, 102, endpoint=True)[1:-1])

    plt.semilogy(w / np.pi / 2, abs(h), label=rf'$\gamma={gamma:.4g}$ $\beta={beta:.4g}$', linestyle=next(LS))


def generate_figure(gamma):
    fig = plt.figure(figsize=(6, 2), dpi=200)
    for i in gamma:
        z_transform(i)
    plt.xlabel(r'frequency [$f_s$]')
    plt.ylabel(r'$|U(z)/A(z)|/\Delta{}t^2$')
    plt.xlim(0, .5)
    plt.grid(which='both')
    plt.legend()

    fig.tight_layout()
    fig.savefig(f'../PIC/Deformation.TF.pdf', format='pdf')


if __name__ == '__main__':
    generate_figure([.5, .8, 1])
