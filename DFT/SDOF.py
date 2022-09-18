import numpy as np
from matplotlib import pyplot as plt

from DFT.FundamentalSolution import mass_proportional

x_max = 10
duration = 10


def compute_response(freq):
    omega_n = 2 * np.pi * freq
    sampling_f = x_max * freq
    samples = int(duration * sampling_f)
    a = .05

    time = np.linspace(0, duration, samples, endpoint=False)
    amplitude = mass_proportional(omega_n, a * omega_n)(time)

    plt.plot(time, amplitude)

    time = np.linspace(0, 10 * duration, 10 * samples, endpoint=False)
    load = np.sin(.1 * omega_n * time)

    plt.plot(time, load)

    plt.plot(time, np.convolve(amplitude, load, 'same'))


if __name__ == '__main__':
    compute_response(1)
    plt.show()
