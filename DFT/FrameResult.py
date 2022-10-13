import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

__SAVE__ = False


def upsample():
    motion = np.loadtxt('../MODEL/FRAME/motion_time')

    ratio = 5
    bin_num = 32 * ratio
    window = signal.windows.nuttall(bin_num + 1)
    cutoff = 1 / ratio
    window *= np.sinc(cutoff * (np.linspace(0, bin_num, bin_num + 1) - bin_num // 2))
    window /= np.sum(window)
    window *= ratio

    length = ratio * len(motion[:, 0])
    up_time = np.linspace(0, length, length, endpoint=False) * motion[1, 0] / ratio
    up_motion = np.zeros(ratio * len(motion[:, 1]))
    up_motion[::ratio] = motion[:, 1]
    up_motion = np.convolve(up_motion, window, mode='same')

    two_column = np.vstack((up_time, up_motion)).T
    np.savetxt('../MODEL/FRAME/up_motion_time', two_column)

    fig = plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(motion[:, 0], motion[:, 1], label='original')
    plt.plot(up_time, up_motion, label='upsampled', linestyle='--', linewidth=.6)
    plt.legend()
    # plt.xlim([24, 44])

    fig.tight_layout()
    plt.show()


def process_result(node):
    fig = plt.figure(figsize=(10, 6), dpi=200)
    fig.add_subplot(211)
    plt.title(f'force history of node {node}')
    plt.xlabel('Time (s)')
    # plt.ylabel(r'Damping Force $F_v$')
    plt.xlim([24, 44])

    name = 'R3-IF'
    dof = 2

    with h5py.File(f'../MODEL/FRAME/{name}.h5', 'r') as f:
        data = f[name][f'{name}{node}']
        time = data[:, 0]
        force = data[:, dof]

    with h5py.File(f'../MODEL/FRAME/up-{name}.h5', 'r') as f:
        data = f[name][f'{name}{node}']
        force_up = data[:, dof]

    plt.plot(time, force, label='original')
    plt.plot(time, force_up, label='upsampled', linestyle='--', linewidth=.6)
    plt.legend()

    fig.add_subplot(212)
    amp = np.fft.rfft(force, 2 * len(force))
    freq = np.fft.rfftfreq(2 * len(amp) - 2, time[1])
    plt.plot(freq, np.abs(amp), label='original')
    plt.plot(freq, np.abs(np.fft.rfft(force_up, 2 * len(force))), label='upsampled', linestyle='--', linewidth=.6)
    plt.yscale('log')

    fig.tight_layout()

    if __SAVE__:
        fig.savefig(f'../PIC/InterpolationExample.pdf', format='pdf')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    for i in range(1, 28):
        if i in {1, 10, 19, 28}:
            continue
        process_result(i)
    # upsample()
