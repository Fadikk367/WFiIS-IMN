import numpy as np
import math

def f(v: float) -> float:
    return v


def g(x: float, v: float, alpha: float) -> float:
    return (alpha*(1.0 - x**2)*v - x)


def F(x_n, v_n, x_n_prev, v_n_prev, dt, alpha) -> float:
    return x_n - x_n_prev - dt/2.0 * (f(v_n_prev) + f(v_n))


def G(x_n, v_n, x_n_prev, v_n_prev, dt, alpha) -> float:
    return v_n - v_n_prev - dt/2.0 * (g(x_n_prev, v_n_prev, alpha) + g(x_n, v_n, alpha))



def rk2(x_n, v_n, dt, alpha) -> (float, float):
    k_1x = v_n
    k_1v = alpha*(1.0 - x_n**2)*v_n - x_n

    k_2x = v_n + dt*k_1v
    k_2v = alpha * (1.0 - (x_n + dt*k_1x)**2) * (v_n + dt*k_1v) - (x_n + dt*k_1x)
    
    x_n_next = x_n + dt/2.0 * (k_1x + k_2x)
    v_n_next = v_n + dt/2.0 * (k_1v + k_2v)

    return x_n_next, v_n_next


def trapezoidal(x_n_prev, v_n_prev, dt, alpha, delta=10e-10) -> (float, float):
    x_n, v_n = x_n_prev, v_n_prev
    while True:
        a_11 = 1.0
        a_12 = -dt/2.0
        a_21 = -dt/2.0 * (-2.0*alpha*x_n*v_n - 1.0)
        a_22 = 1.0 - dt/2.0*alpha*(1.0 - x_n**2)

        dx = (-F(x_n, v_n, x_n_prev, v_n_prev, dt, alpha)*a_22 + G(x_n, v_n, x_n_prev, v_n_prev, dt, alpha)*a_12) / (a_11*a_22 - a_12*a_21)
        dv = (-G(x_n, v_n, x_n_prev, v_n_prev, dt, alpha)*a_11 + F(x_n, v_n, x_n_prev, v_n_prev, dt, alpha)*a_21) / (a_11*a_22 - a_12*a_21)

        x_n += dx
        v_n += dv

        if abs(dx) < delta and abs(dv) < delta:
            break
    
    x_n_next = x_n_prev + dt/2.0*(f(v_n_prev) + f(v_n))
    v_n_next = v_n_prev + dt/2.0*(g(x_n_prev, v_n_prev, alpha) + g(x_n, v_n, alpha))

    return x_n_next, v_n_next


class Solver:
    def __init__(self, fun, constraints, params):
        self.numerical_recipe = fun
        self.constraints = constraints
        self.params = params

    def calculate_error_constants(self, var_2, var_1) -> float:
        return (var_2 - var_1) / (np.exp2(self.params.p) - 1.0)

    def solve(self, output_filename):
        # grid_param = []
        output_file = open(f'./data/{output_filename}', 'w+')

        # for param_set in grid_param:

        dt = self.params.dt_0
        t = self.params.t_0
        x_n = self.params.x_0
        v_n = self.params.v_0

        iter_count = 0
        while True:

            x_n1_2, v_n1_2 = self.numerical_recipe(x_n, v_n, dt, self.params.alpha)
            x_n2_2, v_n2_2 = self.numerical_recipe(x_n1_2, v_n1_2, dt, self.params.alpha)

            x_n2_1, v_n2_1 = self.numerical_recipe(x_n, v_n, 2.0*dt, self.params.alpha)

            E_x = self.calculate_error_constants(x_n2_2, x_n2_1) 
            E_v = self.calculate_error_constants(v_n2_2, v_n2_1) 

            if max(abs(E_x), abs(E_v)) < self.constraints.TOLERANCE:
                t += 2*dt
                x_n = x_n2_2
                v_n = v_n2_2
                output_file.write(f'{t:.8f}\t{dt:.8f}\t{x_n:.8f}\t{v_n:.8f}\n')
                iter_count = 0

            dt = np.power(self.params.S * self.constraints.TOLERANCE / max(abs(E_x), abs(E_v)), 1.0/(self.params.p + 1.0)) * dt
            iter_count += 1

            if t >= self.constraints.t_max:
                break


class Constraints:
    def __init__(self, t_max, TOLERANCE):
        self.TOLERANCE = TOLERANCE
        self.t_max = t_max

class Params:
    def __init__(self, x_0, v_0, t_0, dt_0, S, p, alpha):
        self.x_0 = x_0
        self.v_0 = v_0
        self.t_0 = t_0
        self.dt_0 = dt_0
        self.S = S
        self.p = p
        self.alpha = alpha


if __name__ == "__main__":
    constraints = Constraints(t_max=40.0, TOLERANCE=0.01)
    params = Params(x_0=0.01, v_0=0.0, t_0=0.0, dt_0=1.0, S=0.75, p=2, alpha=5.0)

    solver1 = Solver(rk2, constraints, params)
    solver2 = Solver(trapezoidal, constraints, params)

    solver1.solve(output_filename='rk2-tol-01.tsv')
    solver2.solve(output_filename='trapezoidal-tol-01.tsv')

    constraints.TOLERANCE = 0.00001

    solver1 = Solver(rk2, constraints, params)
    solver2 = Solver(trapezoidal, constraints, params)

    solver1.solve(output_filename='rk2-tol-00001.tsv')
    solver2.solve(output_filename='trapezoidal-tol-00001.tsv')




