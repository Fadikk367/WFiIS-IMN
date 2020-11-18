import numpy as np
from matplotlib import pyplot as plt

def plot_s_results():

  iterations_1, S_1 = np.loadtxt(f'./data/s_global_0.6.tsv', unpack=True, delimiter='\t')
  iterations_2, S_2 = np.loadtxt(f'./data/s_global_1.0.tsv', unpack=True, delimiter='\t')

  iterations_3, S_3 = np.loadtxt(f'./data/s_local_1.0.tsv', unpack=True, delimiter='\t')
  iterations_4, S_4 = np.loadtxt(f'./data/s_local_1.4.tsv', unpack=True, delimiter='\t')
  iterations_5, S_5 = np.loadtxt(f'./data/s_local_1.8.tsv', unpack=True, delimiter='\t')
  iterations_6, S_6 = np.loadtxt(f'./data/s_local_1.9.tsv', unpack=True, delimiter='\t')


  plt.plot(iterations_1, S_1, label=rf'$\omega_g = 0.6,$ it = {iterations_1[-1]:.0f}')
  plt.plot(iterations_2, S_2, label=rf'$\omega_g = 1.0,$ it = {iterations_2[-1]:.0f}')
  plt.title('Relaksacja globalna')
  plt.ylabel('S')
  plt.xlabel('it')
  plt.xscale('log')
  plt.xlim(left=1)
  plt.grid(ls=':')
  plt.legend()
  plt.savefig(f's_global.png')
  plt.clf()


  plt.plot(iterations_3, S_3, label=rf'$\omega_l = 1.0,$ it = {iterations_3[-1]:.0f}')
  plt.plot(iterations_4, S_4, label=rf'$\omega_l = 1.4,$ it = {iterations_4[-1]:.0f}')
  plt.plot(iterations_5, S_5, label=rf'$\omega_l = 1.8,$ it = {iterations_5[-1]:.0f}')
  plt.plot(iterations_6, S_6, label=rf'$\omega_l = 1.9,$ it = {iterations_6[-1]:.0f}')

  plt.title('Relaksacja lokalna')
  plt.ylabel('S')
  plt.xlabel('it')
  plt.xscale('log')
  plt.xlim(left=1)
  plt.grid(ls=':')
  plt.legend()
  plt.savefig(f's_local.png')
  plt.clf()


def plot_color_map(x, y, V, w_g, plot_type):
  z_min, z_max = np.amin(V), np.amax(V)

  plt.xlabel("x")
  plt.ylabel("y")

  plt.title(rf'{plot_type}(x,y),$ \omega_G={w_g}$', fontweight='bold')
  figure = plt.gcf()
  figure.set_size_inches(12, 8)
  plt.pcolor(x, y, np.transpose(V), cmap='seismic', vmin=z_min, vmax=z_max, shading='auto')
  plt.xlim(left=0.0, right=15.0)
  plt.colorbar()
    
  plt.savefig(f'{plot_type}_global_wg_{w_g}.png')
  plt.close()