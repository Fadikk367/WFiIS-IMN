import numpy as np
import params

from numba import njit
from plotter import plot_color_map, plot_contour_map


@njit
def psi(i: int, j: int, PSI, ZETA) -> float:
  return 0.25 * (PSI[i+1, j] + PSI[i-1, j] + PSI[i, j+1] + PSI[i, j-1] - params.delta * params.delta * ZETA[i, j])


@njit
def zeta(i: int, j: int, PSI, ZETA, omega) -> float:
  result = 0.25 * (ZETA[i+1, j] + ZETA[i-1, j] + ZETA[i, j+1] + ZETA[i, j-1])
  result = result - (omega*params.ro)/(16.0*params.mi) * ((PSI[i, j+1] - PSI[i, j-1])*(ZETA[i+1, j] - ZETA[i-1, j]) - (PSI[i+1, j] - PSI[i-1, j])*(ZETA[i, j+1] - ZETA[i, j-1]))
  return result


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
    PSI[params.n_x, j] = Q_wy/(2.0*params.mi) * (np.power(y, 3)/3.0 - np.power(y, 2)/2.0 * y_ny) + Q_we*np.power(y_jl, 2)*(-y_jl + 3.0*y_ny)/(12.0*params.mi)

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

  ZETA[params.i_l][params.j_l] = 0.5*(ZETA[params.i_l-1][params.j_l] + ZETA[params.i_l][params.j_l-1])


def calculate_error(PSI, ZETA) -> float:
  gamma = 0.0
  j_2 = params.j_l + 2
  for i in np.arange(1, params.n_x):
    gamma += PSI[i+1, j_2] + PSI[i-1, j_2] + PSI[i, j_2+1] + PSI[i, j_2-1] - 4.0*PSI[i, j_2] - params.delta**2 * ZETA[i, j_2]

  return gamma


@njit
def calclate_PSI_deriviates(PSI):
  dxPSI = np.zeros((params.n_x + 1, params.n_y + 1))
  dyPSI = np.zeros((params.n_x + 1, params.n_y + 1))

  for i in np.arange(1, params.n_x):
    for j in np.arange(1, params.n_y):
      if i > params.i_l + 1 or j > params.j_l + 1:
        dxPSI[i, j] = -(PSI[i+1, j] - PSI[i-1, j])/(2.0*params.delta)
        dyPSI[i, j] = (PSI[i, j+1] - PSI[i, j-1])/(2.0*params.delta)

  return dxPSI, dyPSI


@njit
def fill_PSI_and_ZETA_interiors(PSI, ZETA, omega):
  for i in np.arange(1, params.n_x):
    for j in np.arange(1, params.n_y):
      if i > params.i_l or j > params.j_l:
        PSI[i, j] = psi(i, j, PSI, ZETA)
        ZETA[i, j] = zeta(i, j, PSI, ZETA, omega)


@njit
def choose_omega(it: int) -> float:
  return (0.0 if it < 2000 else 1.0)


def solve_navier_stokes(Q_we):
  err_file = open(f'./gamma_{Q_we}.tsv', 'w')
  PSI = np.zeros((params.n_x + 1, params.n_y + 1))
  ZETA = np.zeros((params.n_x + 1, params.n_y + 1))

  # fill obstacle rectangle with nans in order to plot contoures correctly
  PSI[0:params.i_l,0:params.j_l] = np.nan
  ZETA[0:params.i_l,0:params.j_l] = np.nan

  Q_wy = calculate_Q_wy(Q_we)
  apply_PSI_edge_conditions(PSI, Q_we, Q_wy)

  omega = 0.0
  for it in np.arange(1, params.IT_MAX + 1):
    omega = choose_omega(it)

    fill_PSI_and_ZETA_interiors(PSI, ZETA, omega)
    apply_ZETA_edge_conditions(ZETA, PSI, Q_we, Q_wy)

    error = calculate_error(PSI, ZETA)
    err_file.write(f'{it}\t{error}\n')

  v, u = calclate_PSI_deriviates(PSI)
  # we need to get rid off borders for deriviates
  v, u = v[1:-1, 1:-1], u[1:-1, 1:-1]

  x = np.linspace(0.0, (params.n_x+1)*params.delta, params.n_x + 1, endpoint=True)
  y = np.linspace(0.0, (params.n_y+1)*params.delta, params.n_y + 1, endpoint=True)

  plot_color_map(x[1:-1], y[1:-1], np.transpose(v), Q_we, "v")
  plot_color_map(x[1:-1], y[1:-1], np.transpose(u), Q_we, "u")

  plot_contour_map(x, y, np.transpose(PSI), Q_we, "psi")
  plot_contour_map(x, y, np.transpose(ZETA), Q_we, "zeta")