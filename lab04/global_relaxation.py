from params import *
from numba import njit
from common import create_starting_matrix, create_density_matrix, apply_egdge_conditions, calculate_S, is_termination_condition_satisfied
from plotter import plot_color_map
import matplotlib.pyplot as plt
import numpy as np


@njit
def calculate_new_elements(V_n, V_s, D):
  for i in np.arange(1, n_x):
    for j in np.arange(1, n_y):
      V_n[i][j] = 0.25 * (V_s[i+1][j] + V_s[i-1][j] + V_s[i][j+1] + V_s[i][j-1] + np.power(delta, 2)/epsilon * D[i][j])


@njit
def compose_solutions(V_n, V_s, w_g):
  for i in np.arange(0, n_x + 1):
    for j in np.arange(1, n_y):
      V_s[i][j] = (1.0 - w_g) * V_s[i][j] + w_g * V_n[i][j]


@njit
def calculate_error_matrix(V, D):
  V_err = np.zeros((n_x, n_y))

  for i in np.arange(1, n_x):
    for j in np.arange(1, n_y):
      V_err[i][j] = (V[i+1][j] - 2.0*V[i][j] + V[i-1][j])/np.power(delta,2) + (V[i][j+1] - 2.0*V[i][j] + V[i][j-1])/np.power(delta,2) + D[i][j]/epsilon

  return V_err


def global_relaxation(w_g):
  result_s = open(f'./data/s_global_{w_g}.tsv', 'w')

  D = create_density_matrix()
  V_s = create_starting_matrix()
  V_n = np.zeros((n_x + 1, n_y + 1))
    

  S_prev = 0.0
  S_curr = 0.0
  it_count = 0

  while True:
    calculate_new_elements(V_n, V_s, D)
    apply_egdge_conditions(V_n)
    compose_solutions(V_n, V_s, w_g)

    S_prev = S_curr
    S_curr = calculate_S(V_s, D)

    result_s.write(f'{it_count}\t{S_curr}\n')

    if it_count > 0 and is_termination_condition_satisfied(S_curr, S_prev):
      break

    it_count += 1

  print(f'GLOBAL: w_g = {w_g}, iteracje : {it_count}')

  V_err = calculate_error_matrix(V_s, D)

  x = np.linspace(x_min, x_max, n_x + 1, endpoint=True)
  y = np.linspace(y_min, y_max, n_y + 1, endpoint=True)

  plot_color_map(x, y, V_err, w_g, 'Verr')
  plot_color_map(x, y, V_s, w_g, 'V')




if __name__ == "__main__":
  global_relaxation(0.6)
