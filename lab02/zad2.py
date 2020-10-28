import numpy as np

N = 500.0
Betha = 0.001
gamma = 0.1
t_0 = 0.0
t_max = 100.0
dt = 0.1
u_0 = 1.0
TOLERANCE = 0.000001
MAX_ITER_COUNT = 20
alpha = Betha*N - gamma


def calculate_steps_number(t_0: float, t_max: float, dt: float) -> float:
  return int((t_max - t_0)/dt) + 1


def is_within_tolerance(curr, prev) -> bool:
  return (np.abs(curr - prev) < TOLERANCE)


def fun(t: float, u: float) -> float:
  return (Betha*N - gamma)*u - Betha* u**2


def newton_iteration(u_prev: float, u_newton_prev: float) -> float:
  return u_prev - (u_newton_prev - u_prev - dt/2.0 * ((alpha*u_prev - Betha*u_prev**2) + (alpha*u_newton_prev - Betha*u_newton_prev**2)))/(1 - dt/2.0*(alpha - 2.0*Betha*u_newton_prev))


def iteratively_enhance_solution(u_prev: float) -> float:
  u_newton_curr = u_prev
  u_newton_prev = u_prev

  iter_count = 0
  while True:
    u_newton_prev = u_newton_curr
    u_newton_curr = newton_iteration(u_prev, u_newton_prev)
    iter_count += 1
    if iter_count > MAX_ITER_COUNT or is_within_tolerance(u_newton_curr, u_newton_prev):
      break
  
  return u_newton_curr


if __name__ == "__main__":
  result_file = open('./data/newton.tsv', 'w+')

  n = calculate_steps_number(t_0, t_max, dt)
  u = np.arange(n, dtype=np.float)
  z = np.arange(n, dtype=np.float)
  u[0] = u_0
  z[0] = N - 1.0
  result_file.write(f'{0:8f}\t{u[0]:8f}\t{z[0]:8f}\n')

  for i in np.arange(1, n):
    u_newton_curr = iteratively_enhance_solution(u[i-1])
    u[i] = u[i-1] + dt/2.0 * (fun((i-1)*dt, u[i-1]) + fun(i*dt, u_newton_curr))
    z[i] = N - u[i]
    result_file.write(f'{i*dt:8f}\t{u[i]:8f}\t{z[i]:8f}\n')

