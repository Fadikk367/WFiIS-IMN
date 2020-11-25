import numpy as np
from numba import njit
from plotter import plot_color_map
from params import *

@njit
def V_B1(y: float) -> float:
  return np.sin(np.pi * y/y_max)

@njit
def V_B2(x: float) -> float:
  return -np.sin(2*np.pi * x/x_max)

@njit
def V_B3(y: float) -> float:
  return np.sin(np.pi * y/y_max)

@njit
def V_B4(x: float) -> float:
  return np.sin(2*np.pi * x/x_max)


@njit
def calculate_S(V, k) -> float:
  S = 0.0
  denominator = 2*k*delta
  for i in np.arange(0, n_x+1-k, k, dtype=np.int32):
    for j in np.arange(0, n_y+1-k, k, dtype=np.int32):
      S += 0.5*np.power(k*delta, 2) * (np.power((V[i+k][j] - V[i][j] + V[i+k][j+k] - V[i][j+k]) / denominator , 2) + np.power((V[i][j+k] - V[i][j] + V[i+k][j+k] - V[i+k][j]) / denominator , 2))
    
  return S

@njit
def is_termination_condition_satisfied(S_curr, S_prev) -> bool:
  return np.abs((S_curr - S_prev)/S_prev) < TOLERANCE


@njit
def create_starting_matrix():
  V = np.zeros((n_x + 1, n_y + 1))

  for i in np.arange(0, n_x + 1):
    x = i*delta
    V[i][0] = V_B4(x)
    V[i][n_y] = V_B2(x)

  for j in np.arange(0, n_y + 1):
    y = j*delta
    V[0][j] = V_B1(y)
    V[n_x][j] = V_B3(y)

  return V



@njit
def thicken_mesh(V, k):
  for i in np.arange(0, n_x+1-k, k, dtype=np.int32):
    for j in np.arange(0, n_y+1-k, k, dtype=np.int32):
      V[i + int(k/2)][j + int(k/2)] = 0.25 * (V[i][j] + V[i+k][j] + V[i][j+k] + V[i+k][j+k])
      if i != n_x-k:
        V[i+k][j + int(k/2)] = 0.5 * (V[i+k][j] + V[i+k][j+k])
      if j != n_y-k:
        V[i + int(k/2)][j+k] = 0.5 * (V[i][j+k] + V[i+k][j+k])
      if j != 0:
        V[i + int(k/2)][j] = 0.5 * (V[i][j] + V[i+k][j])
      if i != 0:
        V[i][j + int(k/2)] = 0.5 * (V[i][j] + V[i][j+k])


@njit
def calculate_new_elements(V, k):
  for i in np.arange(k, n_x+1-k, k, dtype=np.int32):
    for j in np.arange(k, n_y+1-k, k, dtype=np.int32):
      V[i][j] = 0.25 * (V[i+k][j] + V[i-k][j] + V[i][j+k] + V[i][j-k])


def multi_mesh_relaxation():
  k_s = [16, 8, 4, 2, 1]
  V = create_starting_matrix()

  it = 0
  for k in k_s:
    s_file = open(f'./data/s_k_{k}.tsv', 'w')
    curr_it = 0

    S_curr = 0.0
    S_prev = 0.0

    while True:
      calculate_new_elements(V, k)

      S_prev = S_curr
      S_curr = calculate_S(V, k)
      s_file.write(f'{it}\t{S_curr}\n')

      if curr_it > 0 and is_termination_condition_satisfied(S_curr, S_prev):
        break

      it += 1
      curr_it += 1

    x = np.linspace(x_min, x_max, int(n_x/k) + 1, endpoint=True)
    y = np.linspace(y_min, y_max, int(n_y/k) + 1, endpoint=True)

    # V[0::k, 0::k] - matrix subsampling, takes every k-th (starting from 0 up to the end) element from rows and columns 
    plot_color_map(x, y, V[0::k, 0::k], k, 'V')
    thicken_mesh(V, k)
