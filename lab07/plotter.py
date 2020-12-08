def plot_color_map(x, y, V, Q, plot_type):
  z_min, z_max = np.amin(V), np.amax(V)
  x, y = np.meshgrid(x, y)

  plt.xlabel("x")
  plt.ylabel("y")

  plt.title(rf'{plot_type}(x,y),$ k={Q}$', fontweight='bold')
  figure = plt.gcf()
  figure.set_size_inches(10, 8)
  plt.pcolor(x, y, np.transpose(V), cmap='seismic', vmin=z_min, vmax=z_max, shading='auto')
  # plt.xlim(left=0.0, right=15.0)
  plt.colorbar()
    
  plt.savefig(f'{plot_type}_Q_{Q}.png')
  plt.close()