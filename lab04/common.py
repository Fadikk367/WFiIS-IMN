import numpy as np

from numba import njit
from params import *



@njit
def density_1(x: float, y: float) -> float:
  return np.exp(-np.power((x - 0.35*x_max)/sigma_x, 2) - np.power((y - 0.5*y_max)/sigma_y, 2))


@njit
def density_2(x: float, y: float) -> float:
  return -np.exp(-np.power((x - 0.65*x_max)/sigma_x, 2) - np.power((y - 0.5*y_max)/sigma_y, 2))


@njit
def calculate_S(V, density) -> float:
  S = 0.0
  for i in np.arange(0, n_x):
    for j in np.arange(0, n_y):
      S += delta**2 * (0.5*np.power((V[i+1][j] - V[i][j])/delta, 2) + 0.5*np.power((V[i][j+1] - V[i][j])/delta, 2) - density[i][j]*V[i][j])
    
  return S


def is_termination_condition_satisfied(S_curr, S_prev) -> bool:
  return np.abs((S_curr - S_prev)/S_prev) < TOLERANCE


@njit
def create_starting_matrix():
  V_s = np.zeros((n_x + 1, n_y + 1))

  for i in np.arange(0, n_x + 1):
    V_s[i][0] = V_1
    V_s[i][n_y - 1] = V_2
  
  return V_s


@njit
def create_density_matrix():
  D = np.zeros((n_x + 1, n_y + 1))

  for i in np.arange(0, n_x + 1):
    for j in np.arange(0, n_y + 1):
      x = i*delta
      y = j*delta
      D[i][j] = density_1(x, y) + density_2(x, y)

  return D


@njit
def apply_egdge_conditions(V):
  for j in np.arange(1, n_y+1):
    V[0][j] = V[1][j]
    V[n_x][j] = V[n_x - 1][j]
