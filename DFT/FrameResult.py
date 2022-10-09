import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

__SAVE__ = False


def upsample():
    motion = np.loadtxt('../MODEL/FRAME/motion_time')

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

    two_column = np.vstack((up_time, up_motion)).T
    np.savetxt('../MODEL/FRAME/up_motion_time', two_column)


def process_result(node):
    fig = plt.figure(figsize=(10, 6), dpi=200)
    plt.title(f'damping force history of node {node}')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Damping Force $F_v$')
    plt.xlim([20, 60])

    name = 'R3-GDF'

    with h5py.File(f'../MODEL/FRAME/{name}.h5', 'r') as f:
        data = f[name][f'{name}{node}']
        time = data[:, 0]
        force = data[:, 1]

    with h5py.File(f'../MODEL/FRAME/up-{name}.h5', 'r') as f:
        data = f[name][f'{name}{node}']
        force_up = data[:, 1]

    plt.plot(time, np.abs(force - force_up) / np.max(np.abs(force_up)) * 100, label='original')
    plt.legend()

    fig.tight_layout()

    if __SAVE__:
        fig.savefig(f'../PIC/InterpolationExample.pdf', format='pdf')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    process_result(2)
    # upsample()
