import numpy as np
from plotter import plot_color_map


delta = 0.1
nx = 150
nt = 1000
dt = 0.05
XA = 7.5
sigma = 0.5
t_max = dt*nt
XF = 2.5


def extortion(x, t) -> float:
    kronecker_delta = 1 if abs(x - XF) < 1e-3 else 0
    return np.cos(50*t/t_max) * kronecker_delta


def calculate_energy(u, v) -> float:
    E = 0.0
    for i in np.arange(1, nx):
        E += v[i]**2 + ((u[i+1] - u[i-1])/(2*delta))**2

    E *= delta/2.0
    E += delta/4.0 * (((u[1] - u[0])/delta)**2 + ((u[nx] - u[nx-1])/delta)**2)
    return E


def apply_starting_conditions(u, alpha):
    if alpha != 1.0:
        for i in np.arange(1, nx):
            x = delta*i
            u[i] = np.exp(-(x - XA)**2/(2*sigma**2))


def apply_edge_conditions(u, v):
    u[0] = u[nx] = 0.0
    v[0] = v[nx] = 0.0


def calculate_acceleration_vector(a, u, u0, t, alpha, betha):
    for i in np.arange(1, nx):
        x = delta*i
        a[i] = (u[i+1] - 2*u[i] + u[i-1]) / (delta**2) - betha*(u[i] - u0[i])/dt + alpha*extortion(x, t)


def save_u_to_file(u, t, u_file):
    for i in np.arange(nx+1):
        x = delta*i
        u_file.write(f'{t}\t{x}\t{u[i]}\n')




def varlette_schema(alpha, betha):
    E_file = open(f'e_{alpha}_{betha}.tsv', 'w')
    u_file = open(f'u_{alpha}_{betha}.tsv', 'w')

    u0 = np.zeros(nx+1)
    u = np.zeros(nx+1)
    v = np.zeros(nx+1)
    vp = np.zeros(nx+1)
    a = np.zeros(nx+1)

    apply_starting_conditions(u, alpha)
    u0 = u.copy()

    for n in np.arange(nt+1):
        t = dt*n

        vp = v + dt/2.0 * a
        u0 = u.copy()
        u = u + dt*vp
        calculate_acceleration_vector(a, u, u0, t, alpha, betha)
        v = vp + dt/2.0 * a
        E = calculate_energy(u, v)

        save_u_to_file(u, t, u_file)
        E_file.write(f'{t}\t{E}\n')

    E_file.close()
    u_file.close()

    plot_color_map(alpha, betha, f'u_{alpha}_{betha}.tsv')