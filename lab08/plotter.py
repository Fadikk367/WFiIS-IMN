import numpy as np
from matplotlib import pyplot as plt

def plot_linear():
  it00, c0 = np.loadtxt(f'./c_D_0.0.tsv', unpack=True, delimiter='\t')
  it10, c1 = np.loadtxt(f'./c_D_0.1.tsv', unpack=True, delimiter='\t')
  it01, x_sr0 = np.loadtxt(f'./x_D_0.0.tsv', unpack=True, delimiter='\t')
  it11, x_sr1 = np.loadtxt(f'./x_D_0.1.tsv', unpack=True, delimiter='\t')


  plt.plot(it00, c0, label='c, D=0.0')
  plt.plot(it10, c1, label='c, D=0.1')
  plt.plot(it01, x_sr0, label='x_sr, D=0.0')
  plt.plot(it11, x_sr1, label='x_sr, D =0.1')


  plt.title('Zmiany calek')
  plt.ylabel('c, x_sr')
  plt.xlabel('t')
  # plt.xscale('log')
  plt.grid(ls=':')
  plt.legend()
  plt.savefig(f'c_x.png')
  plt.clf()


def plot_color_map(x, y, V, Q, plot_type, fix_min_at_zero=False):
  z_min = 0.0 if fix_min_at_zero else np.amin(V)
  z_max = np.amax(V)

  x, y = np.meshgrid(x, y)

  plt.xlabel("x")
  plt.ylabel("y")

  plt.title(rf'{plot_type}(x,y),$ {Q}$')
  figure = plt.gcf()
  figure.set_size_inches(20, 6)
  plt.pcolor(x, y, V, cmap='plasma', vmin=z_min, vmax=z_max, shading='auto')
  plt.colorbar()
    
  plt.savefig(f'{plot_type}_Q_{Q}.png')
  plt.close()
