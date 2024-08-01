from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import control
from typing import Optional


class triptych_dyn_heatmap(object):

    def __init__(
        self,
        color_map="RdBu",
        interp_type: Optional[str] = "bicubic",
        pane_spacing=0.05,
        tb_margins=0.05,
        lr_margins=0.05,
        cmap_centering=False,
        h_res: int = 500,
        discrete_cmap=False,
        num_discrete_cmap=10,
        shape_def="fig_size",
        shape=(10, 12),
        fig_width_ref=10,
    ):

        self.color_map = color_map
        self.interp_type = interp_type
        self.pane_spacing = pane_spacing
        self.tb_margins = tb_margins
        self.lr_margins = lr_margins
        self.cmap_centering = cmap_centering
        self.h_res = h_res
        self.discrete_cmap = discrete_cmap
        self.num_discrete_cmap = num_discrete_cmap
        self.shape_def = shape_def
        self.shape = shape
        self.fig_width_ref = fig_width_ref
        self.plot_set = [np.zeros((1, 1))] * 3

        return

    def calculate_pane_size(self):
        self.fig_height = self.shape[0]
        self.fig_width = self.shape[1]
        self.pane_height = (
            (1 - (2 * self.tb_margins + 2 * self.pane_spacing)) / 3 * self.fig_height
        )
        self.pane_width = (1 - (2 * self.lr_margins)) * self.fig_width
        self.v_res = int(
            self.h_res
            * (self.pane_height / self.pane_width)
            * (self.fig_width / self.fig_height)
        )
        return

    def calculate_fig_size(self):
        self.pane_width = self.fig_width_ref * (1 - (2 * self.lr_margins))
        self.pane_height = self.pane_width * self.shape[0] / self.shape[1]
        self.fig_width = self.fig_width_ref
        self.fig_height = (3 * self.pane_height) / (
            1 - 2 * (self.tb_margins + self.pane_spacing)
        )
        self.v_res = int(self.h_res * (self.pane_height / self.pane_width))
        return

    def create_dyn_sys(self, z, tf_func):
        #   generates a second order transfer function of a linear mass spring damper system of unit mass and spring constant, given damping coefficient
        #   parameters:
        #   z[in]   - varying parameter, float
        #   tf_func - custom tf function, must be function that accepts a float and returns 2 lists of floats [Num], [Den]
        Num, Den = tf_func(z)
        tf_instance = signal.TransferFunction(Num, Den)
        return tf_instance

    def discretize_plot_set(self):
        for idplot in range(3):
            self.plot_set[idplot] = self.plot_set[idplot] - np.min(
                self.plot_set[idplot]
            )
            step_size = (np.max(self.plot_set[idplot])) / self.num_discrete_cmap
            for idrow, row in enumerate(self.plot_set[idplot]):
                for idcol, element in enumerate(row):
                    snapped_value = np.floor(element / step_size) * step_size
                    self.plot_set[idplot][idrow][idcol] = snapped_value
        return

    def plot_triptic_heatmap(self, plot_name):
        if self.discrete_cmap == True:
            self.discretize_plot_set()

        self.triptic_heatmap = plt.figure(
            plot_name, figsize=(self.fig_width, self.fig_height)
        )
        bottom_0 = round(self.tb_margins, 3)
        bottom_1 = round(
            bottom_0 + self.pane_height / self.fig_height + self.pane_spacing, 3
        )
        bottom_2 = round(
            bottom_1 + self.pane_height / self.fig_height + self.pane_spacing, 3
        )
        bottom = [bottom_2, bottom_1, bottom_0]
        c_scale = [
            np.max(np.abs(self.plot_set[0])),
            np.max(np.abs(self.plot_set[1])),
            np.max(np.abs(self.plot_set[2])),
        ]
        for i in range(3):
            self.triptic_heatmap.add_axes(
                [0, bottom[i], 1, (self.pane_height / self.fig_height)]
            )
            if self.cmap_centering == False:
                plt.imshow(
                    self.plot_set[i],
                    cmap=self.color_map,
                    interpolation=self.interp_type,
                )
            elif self.cmap_centering == True:
                plt.imshow(
                    self.plot_set[i],
                    cmap=self.color_map,
                    interpolation=self.interp_type,
                    vmin=-c_scale[i],
                    vmax=c_scale[i],
                )
            plt.axis("off")

        return


class SOS_triptych_dyn_heatmap(triptych_dyn_heatmap):

    def __init__(
        self,
        dyn_function,
        time_limits=(0.75, 1.46 * 2 * np.pi),
        freq_limits=(0, 3.1),
        damping_coeff_limits=(0.1, 1.3),
        temporal_type="impulse",
        zeroed_start=False,
    ):

        super().__init__()
        self.time_limits = time_limits
        self.freq_limits = freq_limits
        self.damping_coeff_limits = damping_coeff_limits
        self.temporal_type = temporal_type
        self.zeroed_start = zeroed_start
        self.dyn_function = dyn_function

        self.time_step = (self.time_limits[1] - self.time_limits[0]) / self.h_res
        self.time_pad_idx = int(self.time_limits[0] / self.time_step)
        return

    def create_time_phase_damp_linspaces(self):
        # creates sample linspaces for system responses
        # h_res[in]                = total horizontal resolution, int, used for sample time and freq linspace index lengths
        # v_res[in]                = total vertical resolution, int, used for damping coefficient linspace index length
        # time_limits[in]          = number of characteristic oscillations in impulse response, sets end time of impulse response
        # freq_limits[in]
        # damping_coeff_limits[in] = damping coefficient limits, sets start and end of damping coefficients

        self.t_lin = np.linspace(self.time_limits[0], self.time_limits[1], self.h_res)
        self.t_sim = np.linspace(0, self.time_limits[1], self.h_res + self.time_pad_idx)
        self.w_lin = np.linspace(self.freq_limits[0], self.freq_limits[1], self.h_res)
        self.z_lin = np.linspace(
            self.damping_coeff_limits[0], self.damping_coeff_limits[1], self.v_res
        )
        return

    def generate_mag_phase_imp_arrays(self, tf):
        _, mag, phase = signal.bode(tf, self.w_lin)
        if self.temporal_type == "impulse":
            _, imp = signal.impulse(tf, X0=None, T=self.t_sim)
        elif self.temporal_type == "step":
            _, imp = signal.step(tf, X0=None, T=self.t_sim)
        else:
            Exception("invalid time series type, must be impulse (default) or step")
        imp = imp[self.time_pad_idx :]
        if self.zeroed_start == True:
            mag = np.array([mag - mag[0]])
            phase = np.array([phase - phase[0]])
            imp = np.array([imp - imp[0]])
        else:
            mag = np.array([mag])
            phase = np.array([phase])
            imp = np.array([imp])
        return mag, phase, imp

    def sweep_heatmap_arrays(self):
        if self.shape_def == "fig_size":
            self.calculate_pane_size()
        elif self.shape_def == "pane_size":
            self.calculate_fig_size()
        else:
            Exception("imvalid shape_def, must be fig_size or pane_size")

        self.create_time_phase_damp_linspaces()
        for idx, z in enumerate(self.z_lin):
            tf_instance = self.create_dyn_sys(z, self.dyn_function)
            mag, phase, imp = self.generate_mag_phase_imp_arrays(tf_instance)
            if idx == 0:
                mag_set = mag
                phase_set = phase
                imp_set = imp
            else:
                mag_set = np.append(mag_set, mag, axis=0)
                phase_set = np.append(phase_set, phase, axis=0)
                imp_set = np.append(imp_set, imp, axis=0)
        self.data_set = [mag_set, phase_set, imp_set]
        self.plot_set = self.data_set
        return

    def plot_all_line_graph(self, plot_name, loc):
        self.all_line_graph = plt.figure(plot_name)
        idx = int(len(self.plot_set[0]) * loc)
        if idx >= len(self.plot_set[0]):
            idx = len(self.plot_set[0]) - 1
        plt.subplot(3, 1, 1)
        plt.plot(self.w_lin.T, self.plot_set[0][idx])
        plt.subplot(3, 1, 2)
        plt.plot(self.w_lin.T, self.plot_set[1][idx])
        plt.subplot(3, 1, 3)
        plt.plot(self.t_lin.T, self.plot_set[2][idx])
        return


def unit_SOS_tf(z):
    #   mass spring damper TF: H(s) = 1 / (m*s^2 + b*s + k)
    #   w_n = sqrt(k/m)
    #   z   = b / (2*sqrt(km))
    #
    #   define m as 1, w_n as 1, z as input
    #   m = 1
    #   k = m * w_n^2 = 1
    #   b = 2 * z sqrt(w_n * m) = 2 * z
    m = 1
    k = 1
    b = 2 * z
    Num = [1]
    Den = [m, b, k]
    return Num, Den


def two_mass_f1_to_m1_tf(z):

    m1 = 1
    m2 = 1
    k1 = 1
    k2 = 1
    b1 = 2 * z
    b2 = 1 * z
    input = 0
    output = 0

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

    tf = control.ss2tf(A, B[:, input], C[output], D[output, input])

    Num = tf.num[0][0]
    Den = tf.den[0][0]
    return Num, Den


def unit_lpf(z):
    Num = [z]
    Den = [1, z]
    return Num, Den


def unit_hpf(z):
    Num = [1, 0.00001]
    Den = [1, 1 / z]
    return Num, Den


def unit_bpf(z):
    Num = [z, 0.00001]
    Den = [1, 2 * z, 1]
    return Num, Den


def custom_tf_1(z):
    Num = [z, 1]
    Den = [1, z**2, 2]
    return Num, Den
