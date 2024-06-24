import matplotlib
import numpy as np
from matplotlib import colors, pyplot as plt

from DampingForce import get_loc
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
    array_min = max(1e-12, np.min(array))
    array_max = np.max(array)
    array_norm = colors.LogNorm(vmin=array_min, vmax=array_max)
    fig = plt.figure(figsize=(3, 2.4), dpi=400)
    surf = plt.pcolormesh(x, y, np.maximum(1e-12, array).T, norm=array_norm, cmap='RdYlBu', rasterized=True)
    plt.colorbar(surf, aspect=40, ax=plt.gca(), shrink=.9)
    plt.xlabel(r'External Load Frequency $f$ (Hz)')
    plt.ylabel(r'Natural Frequency $f_n$ (Hz)')
    if damping_type == 'Inertial':
        plt.text(0.85, 0.95, rf'$\zeta={a}$', transform=plt.gca().transAxes, ha='center', va='center')
    elif damping_type == 'InertialStiffness':
        plt.text(0.85, 0.95, rf'$a_1={a}$', transform=plt.gca().transAxes, ha='center', va='center')
    elif damping_type == 'InertialMass':
        plt.text(0.85, 0.95, rf'$a_0={a}$', transform=plt.gca().transAxes, ha='center', va='center')
    else:
        raise ValueError('Unknown Damping Type')

    plt.gca().set_aspect('equal')
    fig.tight_layout()
    if __SAVE__:
        fig.savefig(f'../PIC/{damping_type}Map{win_type.capitalize()}_{int(1e5 * a)}.pdf', format='pdf')
        plt.close()
    else:
        plt.title(rf'inertial force with {win_type} window')
        fig.show()


if __name__ == '__main__':
    __SAVE__ = True
    surface('Inertial', .02, 'tri')
    surface('Inertial', .02, 'kaiser')
    surface('Inertial', .02, 'cheb')
    surface('Inertial', .02, 'nuttall')
    surface('InertialStiffness', .0002, 'tri')
    surface('InertialStiffness', .0002, 'hamming')
    surface('InertialStiffness', .0002, 'cheb')
    surface('InertialStiffness', .0002, 'kaiser')
    surface('InertialStiffness', .0002, 'nuttall')
    surface('InertialMass', 2, 'tri')
    surface('InertialMass', 2, 'hamming')
    surface('InertialMass', 2, 'cheb')
    surface('InertialMass', 2, 'kaiser')
    surface('InertialMass', 2, 'nuttall')
