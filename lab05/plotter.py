import numpy as np
from matplotlib import pyplot as plt

def plot_s_results():

  iterations_1, S_1 = np.loadtxt(f'./data/s_k_1.tsv', unpack=True, delimiter='\t')
  iterations_2, S_2 = np.loadtxt(f'./data/s_k_2.tsv', unpack=True, delimiter='\t')
  iterations_3, S_3 = np.loadtxt(f'./data/s_k_4.tsv', unpack=True, delimiter='\t')
  iterations_4, S_4 = np.loadtxt(f'./data/s_k_8.tsv', unpack=True, delimiter='\t')
  iterations_5, S_5 = np.loadtxt(f'./data/s_k_16.tsv', unpack=True, delimiter='\t')

  plt.plot(iterations_5, S_5, label='k = 16')
  plt.plot(iterations_4, S_4, label='k = 8')
  plt.plot(iterations_3, S_3, label='k = 4')
  plt.plot(iterations_2, S_2, label='k = 2')
  plt.plot(iterations_1, S_1, label='k = 1')

  plt.title('Relaksacja multisiatkowa')
  plt.ylabel('S')
  plt.xlabel('it')
  # plt.xscale('log')
  plt.xlim(left=1)
  plt.grid(ls=':')
  plt.legend()
  plt.savefig(f's_multi.png')
  plt.clf()


def plot_color_map(x, y, V, k, plot_type):
  z_min, z_max = np.amin(V), np.amax(V)
  x, y = np.meshgrid(x, y)

  plt.xlabel("x")
  plt.ylabel("y")

  plt.title(rf'{plot_type}(x,y),$ k={k}$', fontweight='bold')
  figure = plt.gcf()
  figure.set_size_inches(10, 8)
  plt.pcolor(x, y, np.transpose(V), cmap='seismic', vmin=z_min, vmax=z_max, shading='auto')
  # plt.xlim(left=0.0, right=15.0)
  plt.colorbar()
    
  plt.savefig(f'{plot_type}_k_{k}.png')
  plt.close()