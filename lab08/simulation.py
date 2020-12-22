import numpy as np
from params import *

from plotter import plot_color_map
from numba import njit


def generate_velocity_field():
    Vx = np.zeros((nx + 1, ny + 1))
    Vy = np.zeros((nx + 1, ny + 1))
    PSI = np.zeros((nx + 1, ny + 1))

    with open('./psi.txt') as f:
        for line in f:
            if (line):
                [i, j, psi] = line.split()
                i, j = int(i), int(j)
                PSI[i, j] = psi

    # Calculate field of velocity
    for i in np.arange(1, nx):
        for j in np.arange(1, ny):
            if not is_obstacle(i, j):
                Vx[i, j] = (PSI[i, j+1] - PSI[i, j-1]) / (2.0*delta)
                Vy[i, j] = -(PSI[i+1, j] - PSI[i-1, j]) / (2.0*delta)

    # Set left and rigth borders
    for j in np.arange(0, ny+1):
        Vx[0, j] = Vx[1, j]
        Vx[nx, j] = Vx[nx-1, j]

    return Vx, Vy


@njit
def calculate_dt(Vx, Vy) -> float:
    v_max = 0.0
    for i in np.arange(1, nx):
        for j in np.arange(1, ny):
            this = np.sqrt(np.power(Vx[i, j], 2) + np.power(Vy[i, j], 2))
            if this > v_max:
                v_max = this

    dt = delta/(4.0*v_max)
    return dt


@njit
def desnity_distribution(x: float, y: float, t = 0.0) -> float:
    return 1.0/(2.0*np.pi*sigma**2) * np.exp(-((x - XA)**2 + (y - YA)**2) / (2.0*sigma**2))


@njit
def is_obstacle(i: int, j: int) -> bool:
    return (i_1 <= i <= i_2) and (0 <= j <= j_1)


@njit
def initialize_matrix(u) -> None:
    for i in np.arange(0, nx+1):
        x = i*delta
        for j in np.arange(1, ny):
            y = j*delta
            if not is_obstacle(i, j):
                u[i, j] = desnity_distribution(x, y)


@njit
def crank_nicolson(u1, u0, Vx, Vy, i, j, dt, D) -> float:
    return (
        (1.0 / (1.0 + 2.0*D*dt/(delta**2))) \
        * (u0[i, j] - dt/2.0 * Vx[i, j] \
        * ((u0[i+1, j] - u0[i-1, j]) / (2.0*delta) + (u1[i+1, j] - u1[i-1, j]) / (2.0*delta)) \
        - dt/2.0 * Vy[i, j] * ((u0[i, j+1] - u0[i, j-1]) / (2.0*delta) + (u1[i, j+1] - u1[i, j-1]) / (2.0*delta)) \
        + dt/2.0 * D * ((u0[i+1, j] + u0[i-1, j] + u0[i, j+1] + u0[i, j-1] - 4*u0[i, j]) / (delta**2) \
        + (u1[i+1, j] + u1[i-1, j] + u1[i, j+1] + u1[i, j-1]) / (delta**2)))
    )


@njit
def crank_nicolson_WB_left(u1, u0, Vx, Vy, i, j, dt, D) -> float:
    return (
        1.0 / (1.0 + 2.0*D*dt/(delta**2)) * (u0[i, j] - dt/2.0 * Vx[i, j] \
        * ((u0[i+1, j] - u0[nx, j]) / (2.0*delta) + (u1[i+1, j] - u1[nx, j]) / (2.0*delta)) \
        - dt/2.0 * Vy[i, j] * ((u0[i, j+1] - u0[i, j-1]) / (2.0*delta) + (u1[i, j+1] - u1[i, j-1]) / (2.0*delta)) \
        + dt/2.0 * D * ((u0[i+1, j] + u0[nx, j] + u0[i, j+1] + u0[i, j-1] - 4*u0[i, j]) / (delta**2) \
        + (u1[i+1, j] + u1[nx, j] + u1[i, j+1] + u1[i, j-1]) / (delta**2)))
    )


@njit
def crank_nicolson_WB_right(u1, u0, Vx, Vy, i, j, dt, D) -> float:
    return (
        1.0 / (1.0 + 2.0*D*dt/(delta**2)) * (u0[i, j] - dt/2.0 * Vx[i, j] \
        * ((u0[0, j] - u0[i-1, j]) / (2.0*delta) + (u1[0, j] - u1[i-1, j]) / (2.0*delta)) \
        - dt/2.0 * Vy[i, j] * ((u0[i, j+1] - u0[i, j-1]) / (2.0*delta) + (u1[i, j+1] - u1[i, j-1]) / (2.0*delta)) \
        + dt/2.0 * D * ((u0[0, j] + u0[i-1, j] + u0[i, j+1] + u0[i, j-1] - 4*u0[i, j]) / (delta**2) \
        + (u1[0, j] + u1[i-1, j] + u1[i, j+1] + u1[i, j-1]) / (delta**2)))
    )


@njit
def calculate_c(u) -> float:   
    c = 0.0
    for i in np.arange(0, nx+1):
        for j in np.arange(0, ny+1):
            c += u[i, j] * delta**2

    return c


@njit
def calculate_x_sr(u) -> float:   
    x_sr = 0.0
    for i in np.arange(0, nx+1):
        x = i*delta
        for j in np.arange(0, ny+1):
            x_sr += x * u[i, j] * delta**2

    return x_sr


@njit
def picard_iteration(u1, u0, Vx, Vy, dt, D):
    for mi in np.arange(1, 20+1):
        for i in np.arange(0, nx+1):
            for j in np.arange(1, ny):
                if (is_obstacle(i, j)):
                    continue
                elif i == 0:
                    u1[i, j] = crank_nicolson_WB_left(u1, u0, Vx, Vy, i, j, dt, D)
                elif i == nx:
                    u1[i, j] = crank_nicolson_WB_right(u1, u0, Vx, Vy, i, j, dt, D)
                else:
                    u1[i, j] = crank_nicolson(u1, u0, Vx, Vy, i, j, dt, D)


def simulate(D):
    x = np.linspace(0.0, (nx+1)*delta, nx+1, endpoint=True)
    y = np.linspace(0.0, (ny+1)*delta, ny+1, endpoint=True)
    c_file = open(f'./c_D_{D}.tsv', 'w')
    x_file = open(f'./x_D_{D}.tsv', 'w')

    Vx, Vy = generate_velocity_field()

    plot_color_map(x, y, np.transpose(Vx), 1, "Vx")
    plot_color_map(x, y, np.transpose(Vy), 1, "Vy")

    dt = calculate_dt(Vx, Vy)

    u0 = np.zeros((nx+1, ny+1))
    u1 = np.zeros((nx+1, ny+1))

    t_max = IT_MAX*delta

    ks = [0, int(IT_MAX/5), 2*int(IT_MAX/5), 3*int(IT_MAX/5), 4*int(IT_MAX/5), IT_MAX]


    initialize_matrix(u0)   

    for it in np.arange(0, IT_MAX+1):

        picard_iteration(u1, u0, Vx, Vy, dt, D)

        np.copyto(u0, u1)


        if it in ks:
            plot_color_map(x, y, np.transpose(u0), f"it={it}, D={D}", "u", True)

        c = calculate_c(u0)
        x_sr = calculate_x_sr(u0)

        c_file.write(f'{dt*it}\t{c}\n')
        x_file.write(f'{dt*it}\t{x_sr}\n')

    c_file.close()
    x_file.close()