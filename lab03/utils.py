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


def trapezoidal(x_n_prev, v_n_prev, dt, alpha, delta=1e-10) -> (float, float):
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