import numpy as np


def morlet(omega_x, omega_y, epsilon=1, sigma=1, omega_0=2):
    return np.exp(-sigma**2 * ((omega_x - omega_0)**2 + (epsilon * omega_y)**2) / 2)


def mexh(omega_x, omega_y, sigma_y=1, sigma_x=1, order=2):
    return -(2 * np.pi) * (omega_x**2 + omega_y**2)**(order / 2) * \
           np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)


def gaus(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1):
    return (1j * omega_x)**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)


def gaus_2(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1):
    return (1j * (omega_x + 1j * omega_y))**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)


def gaus_3(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1, b=1, a=1):
    return (1j * (a * omega_x + b * 1j * omega_y))**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)


def cauchy(omega_x, omega_y, cone_angle=np.pi / 6, sigma=1, l=4, m=4):
    dot1 = np.sin(cone_angle) * omega_x + np.cos(cone_angle) * omega_y
    dot2 = -np.sin(cone_angle) * omega_x + np.cos(cone_angle) * omega_y
    coef = (dot1 ** l) * (dot2 ** m)

    k0 = (l + m) ** 0.5 * (sigma - 1) / sigma
    rad2 = 0.5 * sigma * ((omega_x - k0)**2 + omega_y**2)
    pond = np.tan(cone_angle) * omega_x > abs(omega_y)
    wft = pond * coef * np.exp(-rad2)

    return wft


def dog(omega_x, omega_y, alpha=1.25):
    m = (omega_x**2 + omega_y**2) / 2
    wft = -np.exp(-m) + np.exp(-alpha**2 * m)

    return wft


wavelets = dict(
    morlet=morlet,
    mexh=mexh,
    gaus=gaus,
    gaus_2=gaus_2,
    gaus_3=gaus_3,
    cauchy=cauchy,
    dog=dog
)
