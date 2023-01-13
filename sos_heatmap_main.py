import triptych_dyn_art as tript
import numpy as np
from matplotlib import pyplot as plt
import time

##### --- Parameters --- #####
time_start = time.time()

SOSart = tript.SOS_triptych_dyn_heatmap(tript.unit_SOS_tf)

# SOSart.figure_size   = (9,9)
# SOSart.lr_margins    = 0.01
# SOSart.tb_margins    = 0.01
# SOSart.pane_spacing  = 0.01
# SOSart.interp_type   = None
# SOSart.color_map     = 'tab20'
# SOSart.h_res         = 500

# SOSart.temporal_type = 'step'
# SOSart.damping_coeff_limits = (0.5, 2.5)
# SOSart.num_oscillations = 1
# SOSart.freq_limits = (1,2.5)

SOSart.sweep_heatmap_arrays()
SOSart.plot_triptic_heatmap('heatmap_art')

# SOSart.plot_all_line_graph('line_graph', 0.5)

# time_end = time.time() - time_start
# print(time_end)

plt.show()

