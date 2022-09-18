import numpy as np
from matplotlib import pyplot as plt

from FundamentalSolution import mass_proportional, compute_magnitude

x_max = 10


def compute_response(freq):
    omega_n = 2 * np.pi * freq

    a = .05
    scale = 2 / 3

    duration = -np.log2(1e-2) / a / omega_n

    sampling_f = x_max * freq
    samples = int(duration * sampling_f)

    time = np.linspace(0, duration, samples, endpoint=False)
    amplitude = mass_proportional(omega_n, a * omega_n)(time)

    # plt.plot(time, amplitude)

    time = np.linspace(0, 2 * duration, 2 * samples, endpoint=False)
    load = np.sin(scale * omega_n * time)

    # plt.plot(time, load)
    new_amplitude = np.convolve(amplitude, load, mode='same')
    plt.plot(time, new_amplitude)

    print(np.max(new_amplitude))

    print(compute_magnitude(freq, a, freq * scale))


if __name__ == '__main__':
    fig = plt.figure(figsize=(7, 4), dpi=200)
    compute_response(1.5)
    fig.tight_layout()
    plt.show()
