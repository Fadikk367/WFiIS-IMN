import numpy as np
from matplotlib import pyplot as plt

# Load received results
t_001, y_001, err_001 = np.loadtxt(f'./data/numerical_solution_0.01.tsv', unpack=True, delimiter='\t')
t_01, y_01, err_01 = np.loadtxt(f'./data/numerical_solution_0.1.tsv', unpack=True, delimiter='\t')
t_10, y_10, err_10 = np.loadtxt(f'./data/numerical_solution_1.0.tsv', unpack=True, delimiter='\t')
t, y_analytical = np.loadtxt(f'./data/analytical_solution.tsv', unpack=True, delimiter='\t')


# Plot numerical solutions
plt.plot(t_001, y_001, label='$\Delta t = 0.01$')
plt.plot(t_01, y_01, label='$\Delta t = 0.1$')
plt.plot(t_10, y_10, label='$\Delta t = 1.0$')
plt.plot(t, y_analytical, label='$y_{dok}(t) = e^{\lambda t}$', linestyle='dashed')

plt.title('z.1 - Metoda Eulera - rozwiązanie')
plt.ylabel('y(t)')
plt.xlabel('t')
plt.grid(ls=':')
plt.legend()
plt.savefig(f'zad1_wyniki.png')
plt.clf()


# Global errors
plt.plot(t_001, err_001, label='$\Delta t = 0.01$')
plt.plot(t_01, err_01, label='$\Delta t = 0.1$')
plt.plot(t_10, err_10, label='$\Delta t = 1.0$')

plt.title('z.1 - metoda Eulera - błąd globalny')
plt.ylabel('$y_{num}(t) - y_{dok}(t)$')
plt.xlabel('t')
plt.grid(ls=':')
plt.legend()
plt.savefig('zad1_bledy.png')
plt.clf()