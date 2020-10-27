import numpy as np

N = 500.0
Betha = 0.002
gamma = 0.1
t_0 = 0.0
t_max = 100.0
dt = 0.1
u_0 = 1.0
TOLERANCE = 0.000001


def calculate_steps_number(t_0: float, t_max: float, dt: float) -> float:
  return int((t_max - t_0)/dt) + 1


def is_within_tolerance(curr, prev) -> bool:
  return (np.abs(curr - prev) < TOLERANCE)


def fun(t: float, u: float) -> float:
  return (Betha*N - gamma)*u - Betha* u**2


def analytical_solution(t: float) -> float:
  u_inf = (Betha*N - gamma)/Betha
  u = u_inf/(1 + np.exp(-(gamma - Betha)*(t - t_0)))
  return u



if __name__ == "__main__":
  result_file = open('./data/picard.tsv', 'w+')
  analytical_file = open('./data/picard-anal.tsv', 'w+')

  n = calculate_steps_number(t_0, t_max, dt)
  u = np.arange(n, dtype=np.float)
  u_anal = np.arange(n, dtype=np.float)
  z = np.arange(n, dtype=np.float)
  u[0] = u_0
  u_anal[0] = u_0
  z[0] = N - 1.0
  result_file.write(f'{0.0}\t{u[0]}\t{z[0]}\n')
  analytical_file.write(f'{0.0}\t{u_anal[0]}\t{N - u_anal[0]}\n')

  for i in np.arange(1, n):
    u_prev = u[i-1]
    u_curr = u_prev
    u_anal[i] = analytical_solution(i*dt)
    iter_count = 0
    while True:
      u_prev = u_curr
      alpha = Betha*N - gamma
      # u_curr = u[i-1] + dt/2.0 * ((alpha*u[i-1] - Betha*u[i-1]*u[i-1]) + (alpha*u_prev - Betha*u_prev*u_prev))
      u_curr = u[i-1] + dt/2.0 * (fun((i-1)*dt, u[i-1]) + fun(i*dt, u_prev))
      iter_count += 1
      # print(f'u_prev = {u_prev}')
      # print(f'u[{i}]_{iter_count} = {u_curr}')
      if iter_count > 20 or is_within_tolerance(u_curr, u_prev):
        break

    u[i] = u[i-1] + dt/2.0 * (fun((i-1)*dt, u[i-1]) + fun(i*dt, u_curr))
    z[i] = N - u[i]
    result_file.write(f'{i*dt}\t{u[i]}\t{z[i]}\n')
    analytical_file.write(f'{i*dt}\t{u_anal[i]}\t{N - u_anal[i]}\n')
