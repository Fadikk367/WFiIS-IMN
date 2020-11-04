import numpy as np
import math


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


class Solver:
    def __init__(self, fun, constraints: Constraints, params: Params):
        self.numerical_recipe = fun
        self.constraints = constraints
        self.params = params

    def __calculate_error_constants(self, var_2, var_1) -> float:
        return (var_2 - var_1) / (np.exp2(self.params.p) - 1.0)

    def __adjust_dt(self, prev_dt: float, E_x: float, E_v: float) -> float:
        return np.power(self.params.S * self.constraints.TOLERANCE / max(abs(E_x), abs(E_v)), 1.0/(self.params.p + 1.0)) * prev_dt

    def solve(self, output_filename):
        output_file = open(f'./{output_filename}', 'w+')

        dt = self.params.dt_0
        t = self.params.t_0
        x_n = self.params.x_0
        v_n = self.params.v_0

        while True:
            x_n1_2, v_n1_2 = self.numerical_recipe(x_n, v_n, dt, self.params.alpha)
            x_n2_2, v_n2_2 = self.numerical_recipe(x_n1_2, v_n1_2, dt, self.params.alpha)

            x_n2_1, v_n2_1 = self.numerical_recipe(x_n, v_n, 2.0*dt, self.params.alpha)

            E_x = self.__calculate_error_constants(x_n2_2, x_n2_1) 
            E_v = self.__calculate_error_constants(v_n2_2, v_n2_1) 

            if max(abs(E_x), abs(E_v)) < self.constraints.TOLERANCE:
                t += 2*dt
                x_n = x_n2_2
                v_n = v_n2_2
                output_file.write(f'{t:.8f}\t{dt:.8f}\t{x_n:.8f}\t{v_n:.8f}\n')

            dt = self.__adjust_dt(dt, E_x, E_v)

            if t >= self.constraints.t_max:
                break