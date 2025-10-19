import src.triptych_dyn_art_funcs as tript
import numpy as np
from matplotlib import pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap

##### --- Parameters --- #####
time_start = time.time()


SOSart = tript.SOS_triptych_dyn_heatmap(tript.unit_SOS_tf)

# Color map definition
# colors = [
#     (0.0, "green"),  # at the low end
#     (0.5, "white"),  # midpoint
#     (1.0, "orange"),  # at the high end
# ]
colors = [
    (0.0, "#2D7432"),  # at the low end
    (0.475, "#F4FCDB"),  # midpoint
    (1.0, "#F37521"),  # at the high end
]
color_map = LinearSegmentedColormap.from_list("color_map", colors)

# SOSart.shape         = (9,9)
SOSart.lr_margins = 0
SOSart.tb_margins = 0
SOSart.pane_spacing = 0.02
# SOSart.interp_type   = None
# SOSart.color_map     = 'gray'
SOSart.color_map = color_map


SOSart.h_res = 1000
SOSart.shape_def = "pane_size"
SOSart.shape = (1, 3)
SOSart.discrete_cmap = True
SOSart.num_discrete_cmap = 31
SOSart.interp_type = None

SOSart.damping_coeff_limits = (0.1, 1.3)
# SOSart.time_limits = (0, 8.5)
# SOSart.freq_limits = (0.0, 4)

#### generate heatmaps ####
SOSart.sweep_heatmap_arrays()

#### Plot art ####
SOSart.plot_triptic_heatmap("heatmap_art")
#### Show individual slice  ####
# SOSart.plot_all_line_graph('line_graph', 0.1)

time_end = time.time() - time_start
print("total time: ", round(time_end, 2))

plt.show()
