import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 6})

__SAVE__ = True


def process_result():
    fig = plt.figure(figsize=(6, 6), dpi=200)
    fig.add_subplot(411)
    plt.title(rf'damping force history of the SDOF system')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Damping Force $F_v$')
    with h5py.File('../MODEL/PureSine/NUTTALL.DF.h5', 'r') as f:
        data = f['R2-DF']['R2-DF2']
        plt.plot(data[:, 0], data[:, 1], linewidth=1)
        plt.xlim([0, .5])

    with h5py.File('../MODEL/PureSine/Analytical.h5', 'r') as f:
        data = f['R2-DF']['R2-DF2']
        plt.plot(data[:, 0], data[:, 1], linewidth=2, linestyle='--', c='#e41a1c')

    plt.grid(True, which='both')
    plt.legend(['interpolated external load', 'analytical external load'], loc='upper right')

    fig.add_subplot(412)
    plt.title('frequency response of damping force of the SDOF system')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Magnitude of Damping Force $F_d$')

    with h5py.File('../MODEL/PureSine/NUTTALL.DF.h5', 'r') as f:
        data = f['R2-DF']['R2-DF2']
        o_amplitude = 2 * np.fft.rfft(data[:, 1]) / len(data[:, 1])
        o_freq = np.fft.rfftfreq(2 * len(o_amplitude) - 2, data[1, 0] - data[0, 0])

        plt.plot(o_freq, np.abs(o_amplitude))

        plt.xlim([0, np.max(o_freq)])

    with h5py.File('../MODEL/PureSine/Analytical.h5', 'r') as f:
        data = f['R2-DF']['R2-DF2']
        o_amplitude = 2 * np.fft.rfft(data[:, 1]) / len(data[:, 1])
        o_freq = np.fft.rfftfreq(2 * len(o_amplitude) - 2, data[1, 0] - data[0, 0])

        plt.plot(o_freq, np.abs(o_amplitude), linewidth=2, linestyle='--', c='#e41a1c')

    plt.legend(['interpolated external load', 'analytical external load'])
    plt.grid(True, which='major')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([10, np.max(o_freq)])

    fig.add_subplot(413)
    plt.title(rf'inertial force history of the SDOF system')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Inertial Force $F_a$')
    with h5py.File('../MODEL/PureSine/NUTTALL.IF.h5', 'r') as f:
        data = f['R3-IF']['R3-IF2']
        plt.plot(data[:, 0], data[:, 1], linewidth=1)
        plt.xlim([0, .5])

    with h5py.File('../MODEL/PureSine/IFA.h5', 'r') as f:
        data = f['R3-IF']['R3-IF2']
        plt.plot(data[:, 0], data[:, 1], linewidth=2, linestyle='--', c='#e41a1c')

    plt.grid(True, which='both')
    plt.legend(['interpolated external load', 'analytical external load'], loc='upper right')

    fig.add_subplot(414)
    plt.title('frequency response of inertial force of the SDOF system')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Magnitude of Inertial Force $F_a$')

    with h5py.File('../MODEL/PureSine/NUTTALL.IF.h5', 'r') as f:
        data = f['R3-IF']['R3-IF2']
        o_amplitude = 2 * np.fft.rfft(data[:, 1]) / len(data[:, 1])
        o_freq = np.fft.rfftfreq(2 * len(o_amplitude) - 2, data[1, 0] - data[0, 0])

        plt.plot(o_freq, np.abs(o_amplitude))

        plt.xlim([0, np.max(o_freq)])

    with h5py.File('../MODEL/PureSine/IFA.h5', 'r') as f:
        data = f['R3-IF']['R3-IF2']
        o_amplitude = 2 * np.fft.rfft(data[:, 1]) / len(data[:, 1])
        o_freq = np.fft.rfftfreq(2 * len(o_amplitude) - 2, data[1, 0] - data[0, 0])

        plt.plot(o_freq, np.abs(o_amplitude), linewidth=2, linestyle='--', c='#e41a1c')

    plt.legend(['interpolated external load', 'analytical external load'], loc='upper right')
    plt.grid(True, which='major')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([10, np.max(o_freq)])

    fig.tight_layout()

    if __SAVE__:
        fig.savefig(f'../PIC/Nuttall_Example.pdf', format='pdf')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    process_result()
