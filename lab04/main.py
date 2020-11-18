from local_relaxation import local_relaxation
from global_relaxation import global_relaxation
from plotter import plot_s_results

if __name__ == '__main__':
  w_gs = [0.6, 1.0]
  w_ls = [1.0, 1.4, 1.8, 1.9]

  # GLOBAL RELAXATION
  for w_g in w_gs:
    global_relaxation(w_g)

  # LOCAL RELAXATION
  for w_l in w_ls:
    local_relaxation(w_l)

  plot_s_results()





