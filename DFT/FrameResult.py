import h5py
import numpy as np
from matplotlib import pyplot as plt

__SAVE__ = False


def process_result(node):
    fig = plt.figure(figsize=(6, 6), dpi=200)
    fig.add_subplot(211)
    plt.title(f'damping force history of node {node}')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Damping Force $F_v$')
    with h5py.File('../MODEL/FRAME/R2-IF.h5', 'r') as f:
        data = f['R2-IF'][f'R2-IF{node}']
        plt.plot(data[:, 0], data[:, 1], linewidth=1)
        plt.xlim([20, np.max(data[:, 0])])

    plt.grid(True, which='both')

    fig.add_subplot(212)
    plt.title(f'frequency response of damping force of node {node}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Magnitude of Damping Force $F_d$')

    with h5py.File('../MODEL/FRAME/R2-IF.h5', 'r') as f:
        data = f['R2-IF'][f'R2-IF{node}']
        o_amplitude = 2 * np.fft.rfft(data[:, 1]) / len(data[:, 1])
        o_freq = np.fft.rfftfreq(2 * len(o_amplitude) - 2, data[1, 0] - data[0, 0])
        abs_mag = np.abs(o_amplitude)
        abs_mag /= np.max(abs_mag)
        plt.plot(o_freq, abs_mag)

        plt.xlim([0, np.max(o_freq)])
        plt.ylim([np.min(abs_mag), 1])

    plt.grid(True, which='both')
    plt.yscale('log')

    fig.tight_layout()

    if __SAVE__:
        fig.savefig(f'../PIC/InterpolationExample.pdf', format='pdf')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    for i in range(1, 13):
        if i in {1, 5, 9}:
            continue
        process_result(i)
