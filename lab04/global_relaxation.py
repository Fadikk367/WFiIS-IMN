from params import *
from numba import njit
from common import create_starting_matrix, create_density_matrix, apply_egdge_conditions, calculate_S, is_termination_condition_satisfied
import numpy as np


@njit
def calculate_new_elements(V_n, V_s, D) -> None:
  for i in np.arange(1, n_y-1):
    for j in np.arange(1, n_x-1):
      V_n[i][j] = 0.25 * (V_s[i+1][j] + V_s[i-1][j] + V_s[i][j+1] + V_s[i][j-1] + np.power(delta, 2)/epsilon * D[i][j])


@njit
def compose_solutions(V_n, V_s, w_g):
  V_s *= (1.0 - w_g)
  V_s += w_g*V_n


def global_relaxation(w_g):
  result_s = open(f'./data/s_global_{w_g}.tsv', 'w')
  # result_err = open(f'./err_local_{w_l}.tsv', 'w')

  D = create_density_matrix()
  V_s = create_starting_matrix()
  V_n = np.zeros((n_y, n_x))
    

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

    it_count += 1

    if it_count > 1 and is_termination_condition_satisfied(S_curr, S_prev):
      break

  print(f'GLOBAL: w_g = {w_g}, iteracje : {it_count}')