import numpy as np

l = -1.0
t_min = 0.0
t_max = 5.0

def calc_step_number(dt: float) -> int:
  return int((t_max - t_min)/dt + 1)


def fun(t: float) -> float:
  return np.exp(l*t)


if __name__ == '__main__':
  param_grid = [1.0, 0.1, 0.01]

  for dt in param_grid:
    res = open(f'./data/numerical_solution_{dt}.tsv', 'w+')
    sol = open('./data/analytical_solution.tsv', 'w+')
    n = calc_step_number(dt)

    times = np.empty(n, dtype=np.float)
    values = np.empty(n, dtype=np.float)
    times[0] = 0.0
    values[0] = 1.0

    res.write(f'{times[0]:.8f}\t{values[0]:.8f}\t{(values[0] - fun(0.0)):.8f}\n')
    sol.write(f'{times[0]:.8f}\t{fun(0.0):.8f}\n')

    for i in np.arange(1, n):
      times[i] = i * dt
      values[i] = values[i - 1] + dt*l*values[i - 1]

      res.write(f'{times[i]:.8f}\t{values[i]:.8f}\t{(values[i] - fun(i*dt)):.8f}\n')
      sol.write(f'{times[i]:.8f}\t{fun(i*dt):.8f}\n')

