import matplotlib
import numpy as np
from matplotlib import colors, pyplot as plt

from DFT.DampingForce import get_loc
from FundamentalSolution import compute_response, get_line_style
from PureSine import get_window, ratio, sampling_f

matplotlib.rcParams.update({'font.size': 6})

__LOG_SCALE__ = True

__LOC__ = get_loc()

LS = get_line_style()

__SAVE__ = True


def surface(damping_type, a, win_type: str = 'tri'):
    _, freq, window_amp = get_window(ratio * sampling_f // 2, True, win_type)
    x, y = np.meshgrid(freq, freq)

    array = np.zeros((len(freq), len(freq)))
    for i in range(len(window_amp)):
        _, amp = compute_response(damping_type, a, max(float(i), .01))
        array[:, i] = np.abs(amp * window_amp)

    array /= np.max(array)
    array_min = max(1e-14, np.min(array))
    array_max = np.max(array)
    array_norm = colors.LogNorm(vmin=array_min, vmax=array_max)
    fig = plt.figure(figsize=(3, 2.8), dpi=400)
    surf = plt.pcolormesh(x, y, np.maximum(1e-14, array).T, norm=array_norm, cmap='RdYlBu', rasterized=True)
    plt.colorbar(surf, aspect=40, ax=plt.gca(), shrink=.75)
    plt.xlabel(r'External Load Frequency $\omega$ (Hz)')
    plt.ylabel(r'Natural Frequency $\omega_n$ (Hz)')
    plt.gca().set_aspect('equal')
    fig.tight_layout()
    if __SAVE__:
        fig.savefig(f'../PIC/{damping_type}Map{win_type.capitalize()}-{int(1e5 * a)}.pdf', format='pdf')
    else:
        plt.title(rf'inertial force with {win_type} window')
        fig.show()


if __name__ == '__main__':
    __SAVE__ = False
    surface('InertialStiffness', .02, 'tri')
    surface('InertialStiffness', .02, 'hamming')
    surface('InertialStiffness', .02, 'cheb')
    surface('InertialStiffness', .02, 'kaiser')
