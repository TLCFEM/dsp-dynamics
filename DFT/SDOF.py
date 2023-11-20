import h5py as h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from FundamentalSolution import get_line_style

matplotlib.rcParams.update({'font.size': 6})

__SAVE__ = True

LS = get_line_style()


def process_result():
    fig = plt.figure(figsize=(6, 3), dpi=200)

    fig.add_subplot(211)
    plt.title(rf'inertial force history of the SDOF system')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Inertial Force $F_a$')
    plt.xlim([0, .4])
    for x in ('NEW', 'BATHE', 'GSSSS'):
        with h5py.File(f'../MODEL/PureSine/{x}.IF.h5', 'r') as f:
            data = f['R1-IF']['R1-IF2']
            plt.plot(data[:, 0], data[:, 1], linewidth=1, linestyle=next(LS))

    plt.grid(True, which='both')
    plt.legend([r'Newmark, $\gamma=0.5$, $\beta=0.25$', r'Bathe two-step, $\rho_\infty=0.5$',
                r'GSSSS optimal, $\rho_\infty=0.5$'], loc='upper right')

    fig.add_subplot(212)
    plt.title('frequency response of inertial force of the SDOF system')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Magnitude of Inertial Force $F_a$')

    for x in ('NEW', 'BATHE', 'GSSSS'):
        with h5py.File(f'../MODEL/PureSine/{x}.IF.h5', 'r') as f:
            data = f['R1-IF']['R1-IF2']
            o_amplitude = 2 * np.fft.rfft(data[:, 1]) / len(data[:, 1])
            o_freq = np.fft.rfftfreq(2 * len(o_amplitude) - 2, data[1, 0] - data[0, 0])

            plt.plot(o_freq, np.abs(o_amplitude), linestyle=next(LS))

            plt.xlim([0, np.max(o_freq)])

    plt.grid(True, which='both')
    plt.legend([r'Newmark, $\gamma=0.5$, $\beta=0.25$', r'Bathe two-step, $\rho_\infty=0.5$',
                r'GSSSS optimal, $\rho_\infty=0.5$'], loc='upper right')

    fig.tight_layout()

    if __SAVE__:
        fig.savefig(f'../PIC/VARIOUS.ALGO.pdf', format='pdf')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    process_result()
