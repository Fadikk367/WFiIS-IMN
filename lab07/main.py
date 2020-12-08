import numpy as np
from numba import njit
import params
from plotter import plot_color_map


@njit
def psi(i: int, j: int, PSI, ZETA) -> float:
  return 0.25 * (PSI[i+1, j] + PSI[i-1, j] + PSI[i, j+1] + PSI[i, j-1] - params.delta**2 * ZETA[i, j])


@njit
def zeta(i: int, j: int, PSI, ZETA, omega) -> float:
  result = 0.25 * (ZETA[i+1, j] + ZETA[i-1, j] + ZETA[i, j+1] + ZETA[i, j-1])
  result -= omega*params.ro/(16.0*params.mi) * ((PSI[i, j+1] - PSI[i, j-1])*(ZETA[i+1, j] - ZETA[i-1, j]) - (PSI[i+1, j] - PSI[i-1, j])*(ZETA[i, j+1] - ZETA[i, j-1]))
  return result


@njit
def is_edge_A(i: int, j: int) -> bool:
  return i == 0 and ( params.j_l <= j <= params.n_y)


@njit
def is_edge_B(i: int, j: int) -> bool:
  return j == params.n_y


@njit
def is_edge_C(i: int, j: int) -> bool:
  return i == params.n_x


@njit
def is_edge_D(i: int, j: int) -> bool:
  return params.i_l <= i <= params.n_x


@njit
def is_edge_E(i: int, j: int) -> bool:
  return i == params.i_l and (0 <= j <= params.j_l)


@njit
def is_edge_F(i: int, j: int) -> bool:
  return j == params.j_l and (0 <= i <= params.i_l)


@njit
def is_edge(i: int, j: int) -> bool:
  return is_edge_A(i, j) or is_edge_B(i, j) or is_edge_C(i, j) or is_edge_D(i, j) or is_edge_E(i, j) or is_edge_F(i, j)


@njit
def calculate_Q_wy(Q_we) -> float:
  y_jl = params.j_l * params.delta
  y_ny = params.n_y * params.delta

  return Q_we * (np.power(y_ny, 3) - np.power(y_jl, 3) - 3*y_jl*np.power(y_ny, 2) + 3*np.power(y_jl, 2)*y_ny) / np.power(y_ny, 3)


@njit
def apply_PSI_edge_conditions(PSI, Q_we, Q_wy) -> None:
  y_jl = params.j_l * params.delta
  y_ny = params.n_y * params.delta

  # edge A
  for j in np.arange(params.j_l, params.n_y + 1):
    y = j * params.delta
    PSI[0, j] = Q_we/(2*params.mi) * (np.power(y, 3)/3 - np.power(y, 2)/2 * (y_jl + y_ny) + y*y_jl*y_ny)

  # edge C
  for j in np.arange(0, params.n_y + 1):
    y = j * params.delta
    PSI[params.n_x, j] = Q_wy/(2*params.mi) * (np.power(y, 3)/3 - np.power(y, 2) * y_ny) + Q_we*np.power(y_jl, 2)*(-y_jl + 3*y_ny)/(12*params.mi)

  # edge B 
  for i in np.arange(1, params.n_x):
    PSI[i, params.n_y] = PSI[0, params.n_y]

  # edge D 
  for i in np.arange(params.i_l, params.n_x):
    PSI[i, 0] = PSI[0, params.j_l]

  # edge E 
  for j in np.arange(1, params.j_l + 1):
    PSI[params.i_l, j] = PSI[0, params.j_l]

  # edge F 
  for i in np.arange(1, params.i_l + 1):
    PSI[i, params.j_l] = PSI[0, params.j_l]


@njit
def apply_ZETA_edge_conditions(ZETA, PSI, Q_we, Q_wy) -> None:
  y_jl = params.j_l * params.delta
  y_ny = params.n_y * params.delta

  # edge A
  for j in np.arange(params.j_l, params.n_y + 1):
    y = j * params.delta
    ZETA[0, j] = Q_we/(2*params.mi) * (2*y - y_jl - y_ny)

  # edge C
  for j in np.arange(0, params.n_y + 1):
    y = j * params.delta
    ZETA[params.n_x, j] = Q_wy/(2*params.mi) * (2*y - y_ny)

  # edge B
  for i in np.arange(1, params.n_x):
    ZETA[i, params.n_y] = 2.0/params.delta**2 * (PSI[i, params.n_y-1] - PSI[i, params.n_y])

  # edge D
  for i in np.arange(params.i_l + 1, params.n_x):
    ZETA[i, 0] = 2.0/params.delta**2 * (PSI[i, 1] - PSI[i, 0])

  # edge E
  for j in np.arange(1, params.j_l):
    ZETA[params.i_l, j] = 2.0/params.delta**2 * (PSI[params.i_l+1, j] - PSI[params.i_l, j])

  # edge F
  for i in np.arange(1, params.i_l + 1):
    ZETA[i, params.j_l] = 2.0/params.delta**2 * (PSI[i, params.j_l+1] - PSI[i, params.j_l])


def calculate_error(PSI, ZETA) -> float:
  gamma = 0.0
  j_2 = params.j_l + 2
  for i in np.arange(1, params.n_x):
    gamma += PSI[i+1, j_2] + PSI[i-1, j_2] + PSI[i, j_2+1] + PSI[i, j_2-1] - 4*PSI[i, j_2] - params.delta**2 * ZETA[i, j_2]

  return gamma


def calclate_PSI_x_deriviate(PSI, Q_we, Q_wy):
  dPSI = np.zeros((params.n_x + 1, params.n_y + 1))
  y_jl = params.j_l * params.delta
  y_ny = params.n_y * params.delta

  # fill u_we
  for j in np.arange(params.j_l, params.n_y):
    y = j*params.delta
    u_we = Q_we/(2*params.mi) * (y - y_jl)*(y - y_ny)
    dPSI[0, j] = u_we

  # fill u_wy
  for j in np.arange(1, params.n_y):
    y = j*params.delta
    u_wy = Q_wy/(2*params.mi) * (y - y_ny)
    dPSI[params.n_x, j] = u_wy


  for i in np.arange(1, params.n_x):
    for j in np.arange(1, params.n_y):
      dPSI[i,j] = (PSI[i, j+1] - 2*PSI[i, j] + PSI[i, j-1])/params.delta**2

  return dPSI

def calclate_PSI_y_deriviate(PSI, Q_we, Q_wy):
  dPSI = np.zeros((params.n_x + 1, params.n_y + 1))
  y_jl = params.j_l * params.delta
  y_ny = params.n_y * params.delta

  # fill u_we
  for j in np.arange(params.j_l, params.n_y):
    y = j*params.delta
    u_we = Q_we/(2*params.mi) * (y - y_jl)*(y - y_ny)
    dPSI[0, j] = u_we

  # fill u_wy
  for j in np.arange(1, params.n_y):
    y = j*params.delta
    u_wy = Q_wy/(2*params.mi) * (y - y_ny)
    dPSI[params.n_x - 2, j] = u_wy


  for i in np.arange(1, params.n_x):
    for j in np.arange(1, params.n_y):
      dPSI[i, j] = (PSI[i+1, j] - 2*PSI[i, j] + PSI[i-1, j])/params.delta**2

  return dPSI


if __name__ == "__main__":
  Q_grid = [-1000, -4000, 4000]

  for Q_we in Q_grid:
    err_file = open(f'./gamma_{Q_we}.tsv', 'w')
    PSI = np.zeros((params.n_x + 1, params.n_y + 1))
    ZETA = np.zeros((params.n_x + 1, params.n_y + 1))

    Q_wy = calculate_Q_wy(Q_we)

    apply_PSI_edge_conditions(PSI, Q_we, Q_wy)

    omega = 0.0
    for it in np.arange(1, params.IT_MAX):
      if it < 2000:
        omega = 0.0
      else:
        omega = 1.0

      for i in np.arange(1, params.n_x):
        for j in np.arange(1, params.n_y):
          if not is_edge(i, j):
            PSI[i, j] = psi(i, j, PSI, ZETA)
            ZETA[i, j] = zeta(i, j, PSI, ZETA, omega)
      

      apply_ZETA_edge_conditions(ZETA, PSI, Q_we, Q_wy)
      error = calculate_error(PSI, ZETA)
      err_file.write(f'{it}\t{error}\n')

    dxPSI = calclate_PSI_x_deriviate(PSI, Q_we, Q_wy)
    dyPSI = calclate_PSI_y_deriviate(PSI, Q_we, Q_wy)

    x = np.linspace(0.0, (params.n_x + 1)*params.delta, params.n_x + 1, endpoint=True)
    y = np.linspace(0.0, (params.n_y + 1)*params.delta, params.n_y + 1, endpoint=True)

    plot_color_map(x, y, dxPSI, Q_we, "dxPSI")
    plot_color_map(x, y, dyPSI, Q_we, "dyPSI")
