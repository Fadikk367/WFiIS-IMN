from simulation import simulate
from plotter import plot_linear

if __name__ == "__main__":
    Ds = [0.0, 0.1]

    for D in Ds:
        simulate(D)

    plot_linear()