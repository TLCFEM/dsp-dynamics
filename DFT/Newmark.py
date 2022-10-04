import matplotlib.pyplot as plt
import numpy as np


def compute_transfer(k, c, m, dt, gamma, beta, omega):
    A = np.zeros((2, 2), dtype=complex)
    A[0, 0] = (gamma - 1) * dt * k / m
    A[0, 1] = 1 - (1 - gamma) * dt * c / m
    A[1, 0] = 1 - (.5 - beta) * dt * dt * k / m
    A[1, 1] = dt - (.5 - beta) * dt * dt * c / m

    B = np.zeros((2, 2), dtype=complex)
    B[0, 0] = (1 - gamma) * dt / m
    B[0, 1] = gamma * dt / m
    B[1, 0] = (.5 - beta) * dt * dt / m
    B[1, 1] = beta * dt * dt / m

    C = np.zeros((2, 2), dtype=complex)
    C[0, 0] = gamma * dt * k / m
    C[0, 1] = 1 + gamma * dt * c / m
    C[1, 0] = 1 + beta * dt * dt * k / m
    C[1, 1] = beta * dt * dt * c / m

    D = np.exp(omega * dt * 1j)
    E = np.zeros((2, 1), dtype=complex)
    E[0, 0] = 1
    E[1, 0] = D

    F = np.linalg.solve(C * D - A, np.multiply(B, E))

    D = np.exp(-omega * dt * 1j)
    E[1, 0] = D

    G = np.linalg.solve(C * D - A, np.multiply(B, E))

    return F[0, 0] + G[0, 0]


def compute_kernel(k, zeta, omega, omega_n):
    eta = omega / omega_n
    return 1 / k / (1 - eta ** 2 + 2 * zeta * eta * 1j)


if __name__ == '__main__':
    freq = np.linspace(1, 1000, 1000)
    amp = np.zeros_like(freq, dtype=complex)
    amp2 = np.zeros_like(freq, dtype=complex)

    for i, f in enumerate(freq):
        omega = 2 * np.pi * 80
        m = 1
        k = omega ** 2
        zeta = .02
        c = 2 * zeta * omega * m
        dt = 1 / 2000
        gamma = .5
        beta = .25
        amp[i] = compute_transfer(k, c, m, dt, gamma, beta, 2 * np.pi * f)
        amp2[i] = compute_kernel(k, zeta, 2 * np.pi * f, omega)

    plt.plot(freq, np.abs(amp) / np.abs(amp2))
    plt.grid(which='both')
    plt.legend(['Newmark', 'Exact'])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
