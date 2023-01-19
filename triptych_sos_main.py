import triptych_dyn_art_funcs as tript
import numpy as np
from matplotlib import pyplot as plt
import time
import cmocean

##### --- Parameters --- #####
time_start = time.time()

SOSart = tript.SOS_triptych_dyn_heatmap(tript.two_mass_f1_to_m1_tf)

# SOSart.shape         = (9,9)
# SOSart.lr_margins    = 0.01
# SOSart.tb_margins    = 0.01
SOSart.pane_spacing  = 0.02
# SOSart.interp_type   = None
# SOSart.color_map     = 'coolwarm'
# SOSart.color_map     = cmocean.tools.lighten(cmocean.cm.matter, 1.0)

SOSart.h_res             = 2000
SOSart.shape_def         = 'pane_size'
SOSart.shape             = (1,3)
SOSart.discrete_cmap     = True
SOSart.num_discrete_cmap = 12
SOSart.interp_type       = None

# SOSart.temporal_type = 'step'
SOSart.damping_coeff_limits = (0.1, 0.5)
SOSart.time_limits = (0, 6)
SOSart.freq_limits = (0.1,3)

SOSart.sweep_heatmap_arrays()
SOSart.plot_triptic_heatmap('heatmap_art')
# SOSart.plot_all_line_graph('line_graph', 0.3)

time_end = time.time() - time_start
print(time_end)

plt.show()

