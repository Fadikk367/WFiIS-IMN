from params import *
from numba import njit
from common import create_starting_matrix, create_density_matrix, apply_egdge_conditions, calculate_S, is_termination_condition_satisfied
from plotter import plot_color_map
import numpy as np

@njit
def calculate_new_elements(V, D, w_l):
  for i in np.arange(1, n_x):
    for j in np.arange(1, n_y):
      V[i][j] = (1.0 - w_l) * V[i][j] + 0.25 * w_l * (V[i+1][j] + V[i-1][j] + V[i][j+1] + V[i][j-1] + np.power(delta, 2)/epsilon * D[i][j])


def local_relaxation(w_l):
  result_s = open(f'./data/s_local_{w_l}.tsv', 'w')

  D = create_density_matrix()
  V_s = create_starting_matrix()

  S_prev = 0.0
  S_curr = 0.0
  it_count = 0

  while True:
    calculate_new_elements(V_s, D, w_l)
    apply_egdge_conditions(V_s)

    S_prev = S_curr
    S_curr = calculate_S(V_s, D)

    result_s.write(f'{it_count}\t{S_curr}\n')

    if it_count > 0 and is_termination_condition_satisfied(S_curr, S_prev):
      break

    it_count += 1

  print(f'LOCAL: w_l = {w_l}, iteracje : {it_count}')
  

if __name__ == "__main__":
  local_relaxation(1.0)
