import src.triptych_dyn_art_funcs as tript
import numpy as np
from matplotlib import pyplot as plt
import time

##### --- Parameters --- #####
time_start = time.time()


SOSart = tript.SOS_triptych_dyn_heatmap(tript.unit_SOS_tf)

# SOSart.shape         = (9,9)
SOSart.lr_margins = 0
SOSart.tb_margins = 0
SOSart.pane_spacing = 0.02
# SOSart.interp_type   = None
# SOSart.color_map     = 'gray'
# SOSart.color_map     = cmocean.tools.lighten(cmocean.cm.matter, 1.0)

SOSart.h_res = 3000
SOSart.shape_def = "pane_size"
SOSart.shape = (1, 3)
SOSart.discrete_cmap = False
SOSart.num_discrete_cmap = 11
SOSart.interp_type = None

# SOSart.damping_coeff_limits = (0.15, 1)
# SOSart.time_limits = (0, 8.5)
# SOSart.freq_limits = (0.05,2.25)

#### generate heatmaps ####
SOSart.sweep_heatmap_arrays()

#### Plot art ####
SOSart.plot_triptic_heatmap("heatmap_art")
#### Show individual slice  ####
# SOSart.plot_all_line_graph('line_graph', 0.1)

time_end = time.time() - time_start
print("total time: ", round(time_end, 2))

plt.show()
