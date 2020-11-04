from solver import Solver, Constraints, Params
from utils import rk2, trapezoidal
from plotter import plot_results


if __name__ == "__main__":
    constraints = Constraints(t_max=40.0, TOLERANCE=1e-2)
    params = Params(x_0=0.01, v_0=0.0, t_0=0.0, dt_0=1.0, S=0.75, p=2, alpha=5.0)

    solver1 = Solver(rk2, constraints, params)
    solver2 = Solver(trapezoidal, constraints, params)

    solver1.solve(output_filename='rk2-tol-01.tsv')
    solver2.solve(output_filename='trapezoidal-tol-01.tsv')

    constraints.TOLERANCE = 1e-5

    solver1.solve(output_filename='rk2-tol-00001.tsv')
    solver2.solve(output_filename='trapezoidal-tol-00001.tsv')

    plot_results()

