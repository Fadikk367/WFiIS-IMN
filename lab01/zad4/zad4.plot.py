import numpy as np
from matplotlib import pyplot as plt

# Load received results
t_05, q_05, i_05 = np.loadtxt(f'./data/numerical_solution_0.5.tsv', unpack=True, delimiter='\t')
t_08, q_08, i_08 = np.loadtxt(f'./data/numerical_solution_0.8.tsv', unpack=True, delimiter='\t')
t_10, q_10, i_10 = np.loadtxt(f'./data/numerical_solution_1.0.tsv', unpack=True, delimiter='\t')
t_12, q_12, i_12 = np.loadtxt(f'./data/numerical_solution_1.2.tsv', unpack=True, delimiter='\t')


# Plot numerical solutions for Q(t)
plt.plot(t_05, q_05, label='$0.5 \omega _0$')
plt.plot(t_08, q_08, label='$0.8 \omega _0$')
plt.plot(t_10, q_10, label='$1.0 \omega _0$')
plt.plot(t_12, q_12, label='$1.2 \omega _0$')

plt.title('z.4 - MEtoda RK4, I(t)')
plt.ylabel('Q(t)')
plt.xlabel('t')
plt.grid(ls=':')
plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.1))
plt.savefig('zad4_wyniki_Q.png')
plt.clf()


# Plot numerical solutions for I(t)
plt.plot(t_05, i_05, label='$0.5 \omega _0$')
plt.plot(t_08, i_08, label='$0.8 \omega _0$')
plt.plot(t_10, i_10, label='$1.0 \omega _0$')
plt.plot(t_12, i_12, label='$1.2 \omega _0$')

plt.title('z.4 - MEtoda RK4, I(t)')
plt.ylabel('I(t)')
plt.xlabel('t')
plt.grid(ls=':')
plt.legend(loc='lower left')
plt.savefig('zad4_wyniki_I.png')
plt.clf()