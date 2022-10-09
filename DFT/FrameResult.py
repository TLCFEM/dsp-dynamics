import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

__SAVE__ = False


def upsample():
    fig = plt.figure(figsize=(10, 6), dpi=200)
    motion = np.loadtxt('../MODEL/FRAME/motion_time')
    plt.plot(motion[:, 0], motion[:, 1], linewidth=2)

    ratio = 4
    bin_num = 32 * ratio
    window = signal.windows.nuttall(bin_num + 1)
    cutoff = 1 / ratio
    window *= np.sinc(cutoff * (np.linspace(0, bin_num, bin_num + 1) - bin_num // 2))
    window /= np.sum(window)
    window *= ratio

    up_time = np.linspace(0, ratio * len(motion), ratio * len(motion), endpoint=False) * motion[1, 0] / ratio
    up_motion = np.zeros(ratio * len(motion[:, 1]))
    up_motion[::ratio] = motion[:, 1]
    up_motion = np.convolve(up_motion, window, mode='same')
    plt.plot(up_time, up_motion, '--', linewidth=.5)
    plt.xlim([20, 60])

    fig.tight_layout()
    plt.show()
    print(up_time)
    print(up_motion)

    two_column = np.vstack((up_time, up_motion)).T
    np.savetxt('../MODEL/FRAME/up_motion_time', two_column)


def process_result(node):
    fig = plt.figure(figsize=(6, 6), dpi=200)
    fig.add_subplot(211)
    plt.title(f'damping force history of node {node}')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Damping Force $F_v$')
    with h5py.File('../MODEL/FRAME/R2-IF.h5', 'r') as f:
        data = f['R2-IF'][f'R2-IF{node}']
        plt.plot(data[:, 0], data[:, 2], linewidth=1)
        plt.xlim([20, np.max(data[:, 0])])

    plt.grid(True, which='both')

    fig.add_subplot(212)
    plt.title(f'frequency response of damping force of node {node}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Magnitude of Damping Force $F_d$')

    with h5py.File('../MODEL/FRAME/R2-IF.h5', 'r') as f:
        data = f['R2-IF'][f'R2-IF{node}']
        o_amplitude = 2 * np.fft.rfft(data[:, 2]) / len(data[:, 2])
        o_freq = np.fft.rfftfreq(2 * len(o_amplitude) - 2, data[1, 0] - data[0, 0])
        abs_mag = np.abs(o_amplitude)
        abs_mag /= np.max(abs_mag)
        plt.plot(o_freq, abs_mag)

        plt.xlim([0, np.max(o_freq)])
        plt.ylim([np.min(abs_mag), 1])

    plt.minorticks_on()
    plt.grid(True, which='both')
    plt.yscale('log')

    fig.tight_layout()

    if __SAVE__:
        fig.savefig(f'../PIC/InterpolationExample.pdf', format='pdf')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    for i in {2, 6}:
        process_result(i)
    # upsample()
