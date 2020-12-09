from navier_stokes import solve_navier_stokes
from plotter import plot_errors

if __name__ == "__main__":
  Q_grid = [-1000, -4000, 4000]

  for Q_we in Q_grid:
    solve_navier_stokes(Q_we)

  plot_errors()
