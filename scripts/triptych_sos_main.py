# make_art.py
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import src.triptych_dyn_art_funcs as tript


def main():
    time_start = time.time()

    # --- Color map (green -> off-white -> burnt-orange) ---
    colors = [
        (0.0, "#2D7432"),
        (0.475, "#F4FCDB"),
        (1.0, "#F37521"),
    ]
    color_map = LinearSegmentedColormap.from_list("color_map", colors)

    # --- Instantiate art generator (width-driven layout) ---
    SOSart = tript.SOS_Triptych(
        dyn_function=tript.unit_SOS_tf,  # or any of your TFs
        size_mode="width",  # compute sizes from width + aspect
        width_in=24,  # final figure width in inches (print target)
        shape=(1, 3),  # pane aspect h/w
        lr_margins=0.0,
        tb_margins=0.0,
        pane_spacing=0.02,
        h_res=1000,  # horizontal resolution
        discrete_cmap=True,
        num_discrete_cmap=31,
        interp_type=None,  # no smoothing between cells
        color_map=color_map,
        cmap_centering=False,  # or True to force symmetric color scaling per pane
        damping_coeff_limits=(0.1, 1.3),
        # You can tweak time/freq limits if desired:
        # time_limits=(0.0, 8.5),
        # freq_limits=(0.0, 4.0),
    )

    # --- Generate data and plot triptych ---
    SOSart.sweep_heatmap_arrays()
    fig, axes, ims = SOSart.plot_triptych_heatmap("heatmap_art")

    # --- Save print-ready triptych (24 in wide, 600 dpi, lossless) ---
    fig.savefig(
        "triptych_art_24in.tif",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    # (Optional) Save each pane individually at 24 inches wide:
    SOSart.save_all_panes(
        out_pattern="pane_{i}_24in.tif",
        width_in=24,
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )

    # (Optional) Quick screen preview of a single pane with a colorbar:
    fig_p, ax_p, _ = SOSart.plot_pane(0, width_in=8, with_colorbar=True)
    plt.show()

    elapsed = time.time() - time_start
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
