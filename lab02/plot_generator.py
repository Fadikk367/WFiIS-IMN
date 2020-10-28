import numpy as np
from matplotlib import pyplot as plt

# Load received results
t_1, u_1, z_1 = np.loadtxt(f'./data/picard.tsv', unpack=True, delimiter='\t')
t_2, u_2, z_2 = np.loadtxt(f'./data/newton.tsv', unpack=True, delimiter='\t')
t_3, u_3, z_3 = np.loadtxt(f'./data/RK2.tsv', unpack=True, delimiter='\t')

# Picard
plt.plot(t_1, u_1, label='u(t)')
plt.plot(t_1, z_1, label='z(t)')
plt.title('Metoda Picarda')
plt.ylabel('u(t), v(t)')
plt.xlabel('t')
plt.grid(ls=':')
plt.legend()
plt.savefig(f'picard.png')
plt.clf()

# Newton
plt.plot(t_2, u_2, label='u(t)')
plt.plot(t_2, z_2, label='z(t)')
plt.title('Iteracja Newtona')
plt.ylabel('u(t), v(t)')
plt.xlabel('t')
plt.grid(ls=':')
plt.legend()
plt.savefig(f'newton.png')
plt.clf()

# RK2
plt.plot(t_3, u_3, label='u(t)')
plt.plot(t_3, z_3, label='z(t)')
plt.title('Niejawna RK2')
plt.ylabel('u(t), v(t)')
plt.xlabel('t')
plt.grid(ls=':')
plt.legend()
plt.savefig(f'rk2.png')
plt.clf()