import numpy as np

R = 100.0
L = 0.1
C = 0.001
w_0 = 1.0 / np.sqrt(L * C)
T_0 = 2.0*np.pi / w_0
t_min = 0.0
t_max = 4.0 * T_0

Q_0 = 0.0
I_0 = 0.0

dt = 0.0001
n = int((t_max - t_min)/dt + 1)


def voltage(t: float, w_v: float) -> float:
  return 10.0 * np.sin(w_v * t)


if __name__ == '__main__':
  param_grid = [0.5, 0.8, 1.0, 1.2]

  for multiplier in param_grid:
    res = open(f'./data/numerical_solution_{multiplier}.tsv', 'w+')

    w_v = multiplier * w_0

    times = np.empty(n, dtype=np.float)
    Q_values = np.empty(n, dtype=np.float)
    I_values = np.empty(n, dtype=np.float)
    V_values = np.empty(n, dtype=np.float)
    times[0] = 0.0
    Q_values[0] = Q_0
    I_values[0] = I_0
    V_values[0] = voltage(0.0, w_v)

    res.write(f'{times[0]:.8f}\t{Q_values[0]:.8f}\t{I_values[0]:.8f}\n')

    for i in np.arange(1, n):
      kQ_1 = Q_values[i - 1]
      kI_1 = (V_values[i - 1] / L) - (1.0 / (L*C) * Q_values[i - 1]) - (R / L * I_values[i - 1])
      kQ_2 = I_values[i - 1] + dt/2.0 * kI_1
      kI_2 = (V_values[i - 1] / L) - (1.0 / (L*C) * (Q_values[i - 1] + dt/2.0 * kQ_1)) - (R / L * (I_values[i - 1] + dt/2.0 * kI_1))
      kQ_3 = I_values[i - 1] + dt/2.0 * kI_2
      kI_3 = (V_values[i - 1] / L) - (1.0 / (L*C) * (Q_values[i - 1] + dt/2.0 * kQ_2)) - (R / L * (I_values[i - 1] + dt/2.0 * kI_2))
      kQ_4 = I_values[i - 1] + dt*kI_3
      kI_4 = (V_values[i - 1] / L) - (1.0 / (L*C) * (Q_values[i - 1] + dt*kQ_3)) - (R / L * (I_values[i - 1] + dt*kI_3))

      times[i] = i * dt
      Q_values[i] = Q_values[i - 1] + dt/6.0 * (kQ_1 + 2*kQ_2 + 2*kQ_3 + kQ_4)
      I_values[i] = I_values[i - 1] + dt/6.0 * (kI_1 + 2*kI_2 + 2*kI_3 + kI_4)
      V_values[i] = voltage(dt*i, w_v)

      res.write(f'{times[i]:.8f}\t{Q_values[i]:.8f}\t{I_values[i]:.8f}\n')