# src/triptych_dyn_art_funcs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


@dataclass
class TriptychDynHeatmap:
    # Appearance / rendering
    color_map: any = "RdBu"
    interp_type: Optional[str] = "bicubic"
    pane_spacing: float = (
        0.05  # vertical spacing between panes (fraction of fig height)
    )
    tb_margins: float = 0.05  # top + bottom margins (fraction of fig height)
    lr_margins: float = 0.05  # left + right margins (fraction of fig width)
    cmap_centering: bool = False

    # Discretization (optional)
    discrete_cmap: bool = False
    num_discrete_cmap: int = 10

    # Resolution
    h_res: int = 500  # horizontal sample count for signals/heatmap
    v_res: Optional[int] = None  # if None, computed from pane aspect

    # Geometry
    shape: Tuple[float, float] = (1.0, 3.0)  # pane aspect as (height, width)
    size_mode: str = "width"  # one of {'width','fig','pane'}
    width_in: float = 10.0  # used if size_mode='width'
    fig_size_in: Optional[Tuple[float, float]] = None  # used if size_mode='fig' (W,H)
    pane_width_in: Optional[float] = None  # used if size_mode='pane'

    # Computed layout (inches)
    fig_width: float = 0.0
    fig_height: float = 0.0
    pane_width: float = 0.0
    pane_height: float = 0.0

    # Data containers
    plot_set: List[np.ndarray] | None = None
    data_set: List[np.ndarray] | None = None

    # ---------- Geometry helpers ----------
    def _pane_aspect(self) -> float:
        """Return height/width aspect of a single pane."""
        return self.shape[0] / self.shape[1]

    def compute_layout(self):
        """
        Compute figure and pane sizes based on the selected size_mode.
        All margins/spacing are fractional (0..1) of the figure width/height.
        """
        if self.size_mode == "fig":
            if self.fig_size_in is None:
                raise ValueError("fig_size_in must be provided when size_mode='fig'")
            self.fig_width, self.fig_height = self.fig_size_in
            self.pane_width = self.fig_width * (1 - 2 * self.lr_margins)
            self.pane_height = (
                self.fig_height * (1 - 2 * self.tb_margins - 2 * self.pane_spacing)
            ) / 3.0

        elif self.size_mode == "pane":
            if self.pane_width_in is None:
                raise ValueError("pane_width_in must be provided when size_mode='pane'")
            self.pane_width = self.pane_width_in
            self.pane_height = self.pane_width * self._pane_aspect()
            self.fig_width = self.pane_width / (1 - 2 * self.lr_margins)
            # Solve exactly: fig_h * (1 - 2*(tb + spacing)) = 3*pane_h
            self.fig_height = (3 * self.pane_height) / (
                1 - 2 * (self.tb_margins + self.pane_spacing)
            )

        elif self.size_mode == "width":
            self.fig_width = self.width_in
            self.pane_width = self.fig_width * (1 - 2 * self.lr_margins)
            self.pane_height = self.pane_width * self._pane_aspect()
            self.fig_height = (3 * self.pane_height) / (
                1 - 2 * (self.tb_margins + self.pane_spacing)
            )

        else:
            raise ValueError("size_mode must be one of {'width','fig','pane'}")

        if self.v_res is None:
            # Keep pixels roughly square by scaling vertical samples from pane aspect
            self.v_res = int(self.h_res * (self.pane_height / self.pane_width))

        if self.plot_set is None:
            self.plot_set = [np.zeros((self.v_res, self.h_res)) for _ in range(3)]

    # ---------- Discretization ----------
    def discretize_plot_set(self):
        """Optional quantization to a discrete colormap with num_discrete_cmap bins."""
        if not self.discrete_cmap or self.plot_set is None:
            return
        for i in range(3):
            A = self.plot_set[i]
            A = A - A.min()
            maxA = A.max()
            if maxA == 0:
                self.plot_set[i] = A
                continue
            step = maxA / self.num_discrete_cmap
            self.plot_set[i] = np.floor(A / step) * step

    # ---------- Shared pane renderer ----------
    def _render_pane(
        self,
        ax,
        i: int,
        *,
        cmap_centering: bool | None = None,
        interpolation: str | None = None,
        aspect: str = "auto",
    ):
        """Render pane i into an existing axes and return the image artist."""
        if self.plot_set is None:
            raise RuntimeError("plot_set is empty. Call sweep_heatmap_arrays() first.")
        if not (0 <= i < 3):
            raise IndexError("pane index must be 0, 1, or 2.")

        if cmap_centering is None:
            cmap_centering = self.cmap_centering
        if interpolation is None:
            interpolation = self.interp_type

        A = self.plot_set[i]
        if cmap_centering:
            c = float(np.max(np.abs(A))) if A.size else 1.0
            im = ax.imshow(
                A,
                cmap=self.color_map,
                interpolation=interpolation,
                vmin=-c,
                vmax=c,
                aspect=aspect,
            )
        else:
            im = ax.imshow(
                A,
                cmap=self.color_map,
                interpolation=interpolation,
                aspect=aspect,
            )

        ax.axis("off")
        return im

    # ---------- Triptych plotting ----------
    def plot_triptych_heatmap(self, fig_name: str = "heatmap_art"):
        """Full triptych plot; returns (fig, axes_list, images_list)."""
        self.discretize_plot_set()
        fig = plt.figure(fig_name, figsize=(self.fig_width, self.fig_height))
        pane_frac_h = self.pane_height / self.fig_height

        b0 = self.tb_margins
        b1 = b0 + pane_frac_h + self.pane_spacing
        b2 = b1 + pane_frac_h + self.pane_spacing
        bottoms = [b2, b1, b0]  # top, mid, bottom order to match your original intent

        axes, ims = [], []
        for i in range(3):
            ax = fig.add_axes(
                [self.lr_margins, bottoms[i], 1 - 2 * self.lr_margins, pane_frac_h]
            )
            im = self._render_pane(ax, i)
            axes.append(ax)
            ims.append(im)
        return fig, axes, ims

    # ---------- Single-pane plotting ----------
    def _resolve_pane_figsize(
        self, width_in: float | None, height_in: float | None
    ) -> tuple[float, float]:
        """Resolve (W,H) inches for a single-pane figure."""
        ar = self._pane_aspect()  # H/W
        if width_in is not None:
            return float(width_in), float(width_in) * ar
        if height_in is not None:
            return float(height_in) / ar, float(height_in)
        return self.pane_width, self.pane_height

    def plot_pane(
        self,
        i: int,
        fig_name: str | None = None,
        *,
        width_in: float | None = None,
        height_in: float | None = None,
        with_colorbar: bool = False,
        cmap_centering: bool | None = None,
        interpolation: str | None = None,
    ):
        """Standalone figure for pane i; returns (fig, ax, image)."""
        fw, fh = self._resolve_pane_figsize(width_in, height_in)
        fig = plt.figure(fig_name or f"pane_{i}", figsize=(fw, fh))
        ax = fig.add_axes([0, 0, 1, 1])
        im = self._render_pane(
            ax, i, cmap_centering=cmap_centering, interpolation=interpolation
        )

        if with_colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.02)
            fig.colorbar(im, cax=cax)
        return fig, ax, im

    def save_pane(
        self,
        i: int,
        path: str,
        *,
        width_in: float | None = None,
        height_in: float | None = None,
        dpi: int = 600,
        **savefig_kwargs,
    ):
        """Convenience: render + save pane i. Returns saved path."""
        fig, _, _ = self.plot_pane(i, width_in=width_in, height_in=height_in)
        savefig_kwargs.setdefault("bbox_inches", "tight")
        savefig_kwargs.setdefault("pad_inches", 0)
        fig.savefig(path, dpi=dpi, **savefig_kwargs)
        plt.close(fig)
        return path

    def save_all_panes(
        self,
        out_pattern: str = "pane_{i}.tif",
        *,
        width_in: float | None = None,
        height_in: float | None = None,
        dpi: int = 600,
        **savefig_kwargs,
    ):
        """Export all three panes using identical formatting."""
        paths = []
        for i in range(3):
            path = out_pattern.format(i=i)
            paths.append(
                self.save_pane(
                    i,
                    path,
                    width_in=width_in,
                    height_in=height_in,
                    dpi=dpi,
                    **savefig_kwargs,
                )
            )
        return paths


class SOS_Triptych(TriptychDynHeatmap):
    """
    Triptych driven by sweeping damping coefficient for a second-order system (or any custom TF).
    """

    def __init__(
        self,
        dyn_function: Callable[[float], Tuple[list, list]],
        time_limits=(0.75, 1.46 * 2 * np.pi),
        freq_limits=(0.0, 3.1),
        damping_coeff_limits=(0.1, 1.3),
        temporal_type="impulse",
        zeroed_start=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dyn_function = dyn_function
        self.time_limits = time_limits
        self.freq_limits = freq_limits
        self.damping_coeff_limits = damping_coeff_limits
        self.temporal_type = temporal_type
        self.zeroed_start = zeroed_start

        self.time_step = (self.time_limits[1] - self.time_limits[0]) / self.h_res
        self.time_pad_idx = int(self.time_limits[0] / self.time_step)

    @staticmethod
    def create_dyn_sys(z: float, tf_func: Callable[[float], Tuple[list, list]]):
        Num, Den = tf_func(z)
        return signal.TransferFunction(Num, Den)

    def _gen_mag_phase_imp(self, tf):
        w_lin = np.linspace(self.freq_limits[0], self.freq_limits[1], self.h_res)
        _, mag, phase = signal.bode(tf, w_lin)

        t_sim = np.linspace(0, self.time_limits[1], self.h_res + self.time_pad_idx)
        if self.temporal_type == "impulse":
            _, imp = signal.impulse(tf, X0=None, T=t_sim)
        elif self.temporal_type == "step":
            _, imp = signal.step(tf, X0=None, T=t_sim)
        else:
            raise ValueError("temporal_type must be 'impulse' or 'step'")
        imp = imp[self.time_pad_idx :]

        if self.zeroed_start:
            mag = (mag - mag[0])[None, :]
            phase = (phase - phase[0])[None, :]
            imp = (imp - imp[0])[None, :]
        else:
            mag = mag[None, :]
            phase = phase[None, :]
            imp = imp[None, :]
        return mag, phase, imp

    def sweep_heatmap_arrays(self):
        """Compute data_set and plot_set based on geometry and parameter sweep."""
        self.compute_layout()
        z_lin = np.linspace(
            self.damping_coeff_limits[0], self.damping_coeff_limits[1], self.v_res
        )

        mag_set = []
        phase_set = []
        imp_set = []

        for z in z_lin:
            tf = self.create_dyn_sys(z, self.dyn_function)
            mag, phase, imp = self._gen_mag_phase_imp(tf)
            mag_set.append(mag)
            phase_set.append(phase)
            imp_set.append(imp)

        mag_set = np.vstack(mag_set)
        phase_set = np.vstack(phase_set)
        imp_set = np.vstack(imp_set)

        self.data_set = [mag_set, phase_set, imp_set]
        self.plot_set = self.data_set


# --------- Example transfer functions (same as your originals) ---------
def unit_SOS_tf(z: float):
    # mass-spring-damper: H(s) = 1 / (s^2 + 2*z*s + 1) for m=k=1
    m = 1.0
    k = 1.0
    b = 2.0 * z
    return [1.0], [m, b, k]


def two_mass_f1_to_m1_tf(z: float):
    import control

    m1 = 1.0
    m2 = 1.0
    k1 = 1.0
    k2 = 1.0
    b1 = 2.0 * z
    b2 = 1.0 * z
    input_idx = 0
    output_idx = 0

    A = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-(k1 + k2) / m1, k2 / m1, -(b1 + b2) / m1, b2 / m1],
            [k2 / m2, -k2 / m2, b2 / m2, -b2 / m2],
        ]
    )
    B = np.array([[0, 0], [0, 0], [1 / m1, 0], [0, 1 / m2]])
    C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    D = np.array([[0, 0], [0, 0]])

    tf = control.ss2tf(A, B[:, input_idx], C[output_idx], D[output_idx, input_idx])
    Num = tf.num[0][0]
    Den = tf.den[0][0]
    return Num, Den


def unit_lpf(z: float):
    return [z], [1, z]


def unit_hpf(z: float):
    return [1, 0.00001], [1, 1 / z]


def unit_bpf(z: float):
    return [z, 0.00001], [1, 2 * z, 1]


def custom_tf_1(z: float):
    return [z, 1], [1, z**2, 2]
