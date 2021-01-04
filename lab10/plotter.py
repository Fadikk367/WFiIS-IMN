import numpy as np
from matplotlib import pyplot as plt

nx = 150
nt = 1000
dx = 0.1
dt = 0.05

def plot_linear():
    t1, e1 = np.loadtxt(f'./e_0.0_0.0.tsv', unpack=True, delimiter='\t')
    t2, e2 = np.loadtxt(f'./e_0.0_0.1.tsv', unpack=True, delimiter='\t')
    t3, e3 = np.loadtxt(f'./e_0.0_1.0.tsv', unpack=True, delimiter='\t')
    t4, e4 = np.loadtxt(f'./e_1.0_1.0.tsv', unpack=True, delimiter='\t')


    plt.plot(t1, e1, label='E1, B=0.0')
    plt.plot(t2, e2, label='E2, B=0.1')
    plt.plot(t3, e3, label='E3, B=1.0')


    plt.title('Zmiany energii')
    plt.ylabel('E')
    plt.xlabel('t')
    plt.grid(ls=':')
    plt.legend()
    plt.savefig(f'E_1_2_3.png')
    plt.clf()


    plt.plot(t4, e4, label='E, a=1.0, B=1.0')
    plt.title('Zmiana energii')
    plt.ylabel('E')
    plt.xlabel('t')
    plt.grid(ls=':')
    plt.legend()
    plt.savefig(f'E_4.png')
    plt.clf()


def plot_color_map(alpha, beta, filename):
    t, x, u = np.loadtxt(filename, unpack=True, delimiter='\t')

    V = np.zeros((nx+1, nt+1))

    for t in range(1, nt + 1):
        for x in range(nx + 1):
            V[x, t] = u[x+(nx+1)*(t-1)]

    x = np.linspace(0.0, (nx+1)*dx, nx+1, endpoint=True)
    t = np.linspace(0.0, (nt+1)*dt, nt+1, endpoint=True)

    z_min = np.amin(V)
    z_max = np.amax(V)

    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f'u(x,t)')

    figure = plt.gcf()
    figure.set_size_inches(12, 4)
    plt.pcolor(t, x, V, cmap='plasma', vmin=z_min, vmax=z_max, shading='auto')
    plt.colorbar()
      
    plt.savefig(f'u(x,t)_{alpha}_{beta}.png')
    plt.close()