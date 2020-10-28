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

a_11 = 0.25
a_12 = 0.25 - np.sqrt(3)/6.
a_21 = 0.25 + np.sqrt(3)/6.
a_22 = 0.25
b_1 = 0.5 
b_2 = 0.5
c_1 = 0.5 - np.sqrt(3)/6.
c_2 = 0.5 + np.sqrt(3)/6.


def calculate_steps_number(t_0: float, t_max: float, dt: float) -> float:
  return int((t_max - t_0)/dt) + 1


def is_within_tolerance(curr, prev) -> bool:
  return (np.abs(curr - prev) < TOLERANCE)


def fun(t: float, u: float) -> float:
  return (Betha*N - gamma)*u - Betha * u**2


def F_1(U_1: float, U_2: float, u_n: float) -> float:
  return U_1 - u_n - dt*(a_11*(alpha*U_1 - Betha*U_1**2) + a_12*(alpha*U_2 - Betha*U_2**2))


def F_2(U_1: float, U_2: float, u_n: float) -> float:
  return U_2 - u_n - dt*(a_21*(alpha*U_1 - Betha*U_1**2) + a_22*(alpha*U_2 - Betha*U_2**2))


if __name__ == "__main__":
  result_file = open('./data/RK2.tsv', 'w+')

  n = calculate_steps_number(t_0, t_max, dt)
  u = np.arange(n, dtype=np.float)
  z = np.arange(n, dtype=np.float)
  u[0] = u_0
  z[0] = N - u_0
  result_file.write(f'{0.0:.8f}\t{u[0]:.8f}\t{z[0]:.8f}\n')

  for i in np.arange(1, n):
    U_1 = u[i-1]
    U_2 = u[i-1]
    U_1_prev = U_1
    U_2_prev = U_1
    dU_1 = 0.0
    dU_2 = 0.0
    iter_count = 0

    while True:
      m_11 = 1.0 - dt*a_11*(alpha - 2*Betha*U_1_prev)
      m_12 = -dt*a_12*(alpha - 2*Betha*U_2_prev)
      m_21 = -dt*a_21*(alpha - 2*Betha*U_1_prev)
      m_22 = 1.0 - dt*a_22*(alpha - 2*Betha*U_2_prev)

      dU_1 = (F_2(U_1_prev, U_2_prev, u[i-1])*m_12 - F_1(U_1_prev, U_2_prev, u[i-1])*m_22) / (m_11*m_22 - m_12*m_21)
      dU_2 = (F_2(U_1_prev, U_2_prev, u[i-1])*m_21 - F_1(U_1_prev, U_2_prev, u[i-1])*m_11) / (m_11*m_22 - m_12*m_21)

      U_1 = U_1_prev + dU_1
      U_2 = U_2_prev + dU_2

      U_1_prev = U_1
      U_2_prev = U_2

      iter_count += 1
      if iter_count > MAX_ITER_COUNT or is_within_tolerance(U_1, U_1_prev):
        break

    u[i] = u[i-1] + dt * (b_1*fun((i-1)*dt + c_1*dt, U_1) + b_2*fun((i-1)*dt + c_2*dt, U_2))
    z[i] = N - u[i]
    result_file.write(f'{i*dt:.8f}\t{u[i]:.8f}\t{z[i]:.8f}\n')