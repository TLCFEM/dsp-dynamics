import matplotlib.pyplot as plt
import numpy as np

from FundamentalSolution import get_line_style

LS = get_line_style()

__SAVE__ = True


def compute_transfer(k, c, m, h, gamma, beta, f):
    mat_a = np.zeros((2, 2))
    mat_a[0, 0] = (gamma - 1) * h * k / m
    mat_a[0, 1] = 1 + (gamma - 1) * h * c / m
    mat_a[1, 0] = 1 + (beta - .5) * h * h * k / m
    mat_a[1, 1] = h + (beta - .5) * h * h * c / m

    mat_b = np.zeros((2, 2))
    mat_b[0, 0] = (1 - gamma) * h / m
    mat_b[0, 1] = gamma * h / m
    mat_b[1, 0] = (.5 - beta) * h * h / m
    mat_b[1, 1] = beta * h * h / m

    mat_c = np.zeros((2, 2))
    mat_c[0, 0] = gamma * h * k / m
    mat_c[0, 1] = 1 + gamma * h * c / m
    mat_c[1, 0] = 1 + beta * h * h * k / m
    mat_c[1, 1] = beta * h * h * c / m

    scalar_d = np.exp(f * h * 1j)
    mat_e = np.zeros((2, 1), dtype=complex)
    mat_e[0, 0] = 1
    mat_e[1, 0] = scalar_d

    result = np.linalg.solve(mat_c * scalar_d - mat_a, np.matmul(mat_b, mat_e))

    return result[0, 0]


def compute_kernel(k, zeta, omega, omega_n):
    eta = omega / omega_n
    return 1 / k / (1 - eta ** 2 + 2 * zeta * eta * 1j)


def compute_response(omega, freq, dt, gamma, beta):
    m = 1
    k = omega ** 2
    c = 2 * .05 * omega * m
    zeta = c / 2 / np.sqrt(k * m)

    analytical = np.zeros_like(freq, dtype=complex)
    newmark = np.zeros_like(freq, dtype=complex)
    for i, f in enumerate(freq):
        newmark[i] = compute_transfer(k, c, m, dt, gamma, beta, 2 * np.pi * f)
        analytical[i] = compute_kernel(k, zeta, 2 * np.pi * f, omega)

    return newmark / analytical


def generate_figure(gamma, dt=1 / 2000):
    response = {
        1: None,
        2: None,
        5: None,
        10: None,
        20: None,
        50: None,
        100: None,
        200: None,
        500: None
    }

    beta = .25 * (.5 + gamma) ** 2

    freq = np.logspace(0, 3, 1000)

    fig = plt.figure(figsize=(6, 2), dpi=200)
    plt.title(rf'Newmark method with $\gamma={gamma}$ and $\beta={beta:.4g}$')

    for f in response.keys():
        response[f] = compute_response(f * 2 * np.pi, freq, dt, gamma, beta)
        plt.plot(freq, np.abs(response[f]), label=f'$f_n={f}$ Hz', linestyle=next(LS))

    plt.legend(handlelength=3, loc='lower left', ncol=3)
    plt.grid(which='both', linestyle='--', linewidth=.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'$|\hat{h}_{NM}|/|\hat{h}|$')
    plt.xscale('log')
    plt.yscale('log')
    fig.tight_layout()
    if __SAVE__:
        fig.savefig(f'../PIC/Newmark-{gamma}-{dt * 1000}.pdf', format='pdf')
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    generate_figure(.8, 1 / 2000)
    generate_figure(1, 1 / 2000)
    generate_figure(1.5, 1 / 2000)
