import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

sampling_f = 50
ratio = 4


def get_window(length: int = 512, half: bool = False):
    up_sampling_f = sampling_f * ratio

    window = signal.windows.triang(2 * ratio - 1)

    if half:
        window_amp = np.fft.rfft(window, 2 * length)
        window_freq = np.fft.rfftfreq(2 * len(window_amp) - 2, 1 / up_sampling_f)
    else:
        window_amp = np.fft.fftshift(np.fft.fft(window, length))
        window_freq = np.fft.fftshift(np.fft.fftfreq(len(window_amp), 1 / up_sampling_f))

    window_amp /= np.max(np.abs(window_amp))

    return window, window_freq, window_amp


if __name__ == '__main__':
    window, window_freq, window_amp = get_window(500, True)
    window_amp = np.abs(window_amp)
    fig = plt.figure(figsize=(6, 3), dpi=200)
    plt.plot(window_freq, 20 * np.log10(np.maximum(window_amp, 1e-5)))
    fig.tight_layout()
    plt.grid(which='both')
    plt.show()
