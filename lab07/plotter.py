import numpy as np
from matplotlib import pyplot as plt

def plot_errors():
  iterations_1, err_1 = np.loadtxt(f'./gamma_-1000.tsv', unpack=True, delimiter='\t')
  iterations_2, err_2 = np.loadtxt(f'./gamma_-4000.tsv', unpack=True, delimiter='\t')
  iterations_3, err_3 = np.loadtxt(f'./gamma_4000.tsv', unpack=True, delimiter='\t')

  plt.plot(iterations_1, err_1, label='Q = -1000')
  plt.plot(iterations_2, err_2, label='Q = -4000')
  plt.plot(iterations_3, err_3, label='Q = 4000')

  plt.title('Zmiany bledow')
  plt.ylabel('gamma')
  plt.xlabel('it')
  plt.xscale('log')
  plt.grid(ls=':')
  plt.legend()
  plt.savefig(f'errors.png')
  plt.clf()


def plot_color_map(x, y, V, Q, plot_type):
  z_min, z_max = np.amin(V), np.amax(V)
  x, y = np.meshgrid(x, y)

  plt.xlabel("x")
  plt.ylabel("y")

  plt.title(rf'{plot_type}(x,y),$ Q={Q}$')
  figure = plt.gcf()
  figure.set_size_inches(20, 9)
  plt.pcolor(x, y, V, cmap='turbo', vmin=z_min, vmax=z_max, shading='auto')
  plt.colorbar()
    
  plt.savefig(f'{plot_type}_Q_{Q}.png')
  plt.close()

def plot_contour_map(x, y, V, Q, plot_type):
  plt.xlabel("x")
  plt.ylabel("y")

  plt.title(rf'{plot_type}(x,y),$ Q={Q}$')
  figure = plt.gcf()
  figure.set_size_inches(20, 9)
  plt.contour(x, y, V, cmap='copper', levels=40)
  plt.colorbar()
    
  plt.savefig(f'{plot_type}_Q_{Q}.png')
  plt.close()