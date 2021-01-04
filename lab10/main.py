from varlette_schema import varlette_schema
from plotter import plot_linear


if __name__ == '__main__':
    param_grid = [
      { 'alpha': 0.0, 'betha': 0.0 }, 
      { 'alpha': 0.0, 'betha': 0.1 }, 
      { 'alpha': 0.0, 'betha': 1.0 },
      { 'alpha': 1.0, 'betha': 1.0 }
    ]
    
    for params in param_grid:
        varlette_schema(params['alpha'], params['betha'])

    plot_linear()