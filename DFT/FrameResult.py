import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

matplotlib.rcParams.update({'font.size': 6})

__SAVE__ = True


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

    fig = plt.figure(figsize=(6, 2), dpi=200)
    plt.plot(motion[:, 0], motion[:, 1], label='original', marker='o', markevery=5)
    plt.plot(up_time, up_motion, label='upsampled', marker='o', markevery=5, linewidth=.6)
    plt.legend()
    plt.xlim([24, 40])

    fig.tight_layout()
    plt.show()


def process_result(node):
    fig = plt.figure(figsize=(6, 3), dpi=200)
    fig.add_subplot(211)
    plt.title(f'inertial force history of node {node}')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Inertial Force $F_a$')
    plt.xlim([24, 40])

    name = 'R3-IF'
    dof = 2

    with h5py.File(f'../MODEL/FRAME/{name}.h5', 'r') as f:
        data = f[name][f'{name}{node}']
        time = data[:, 0]
        force = data[:, dof]

    with h5py.File(f'../MODEL/FRAME/up-{name}.h5', 'r') as f:
        data = f[name][f'{name}{node}']
        force_up = data[:, dof]

    if np.max(np.abs(force)) > np.max(np.abs(force_up)):
        plt.plot(time, force, label='linear interpolation', linewidth=1)
        plt.plot(time, force_up, '-r', label='Blackman-Nuttall window', linewidth=1)
    else:
        plt.plot(time, force_up, '-r', label='Blackman-Nuttall window', linewidth=1)
        plt.plot(time, force, label='linear interpolation', linewidth=1)
    plt.grid(which='major')
    plt.legend(loc='upper right')

    fig.add_subplot(212)
    plt.title(f'inertial force spectrum of node {node}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Magnitude $|\hat{F_a}|$')
    amp = 2 * np.fft.rfft(force, len(force)) / len(force)
    freq = np.fft.rfftfreq(2 * len(amp) - 2, time[1])
    plt.plot(freq, np.abs(amp), label='linear interpolation', linewidth=1)
    up_amp = 2 * np.fft.rfft(force_up, len(force)) / len(force_up)
    plt.plot(freq, np.abs(up_amp), '-r', label='Blackman-Nuttall window', linewidth=1)
    plt.grid(which='major')
    plt.yscale('log')
    plt.xlim([0, 125])
    plt.legend(loc='upper right')

    fig.tight_layout()

    if __SAVE__:
        fig.savefig(f'../PIC/FrameExample-{node}.pdf', format='pdf')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    for i in {2, 4}:
        process_result(i)
    # upsample()
