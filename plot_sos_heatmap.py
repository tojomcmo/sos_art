from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import impulse as impulseResponse
from scipy.signal import bode as bode
import control as control

class art_definitions(object):

    def __init__(self, figure_size = (12,10), color_map = 'RdBu', interp_type = 'bicubic', pane_spacing = 0.1, tb_spacing = 0.1, lr_spacing = 0.1, zero_centering = False, h_res:int = 500):
        self.figure_size     = figure_size
        self.color_map       = color_map
        self.interp_type     = interp_type
        self.pane_spacing    = pane_spacing
        self.tb_spacing      = tb_spacing
        self.lr_spacing      = lr_spacing
        self.zero_centering  = zero_centering
        self.h_res           = h_res

        self.calculate_pane_size()
        return

    def calculate_pane_size(self):
        self.pane_height     = (1 - (2 * self.tb_spacing + 2 * self.pane_spacing))/3
        self.pane_width      = 1 - (2 * self.lr_spacing)
        self.v_res           = int(self.h_res * (self.pane_height/self.pane_width) * (self.figure_size[1]/self.figure_size[0]))
        return

    def set_tb_spacing(self, new_spacing:float):
        self.tb_spacing = new_spacing
        self.calculate_pane_size()  

    def set_lr_spacing(self, new_spacing:float):
        self.lr_spacing = new_spacing
        self.calculate_pane_size()    

    def set_pane_spacing(self, new_spacing:float):
        self.pane_spacing = new_spacing
        self.calculate_pane_size() 

    def set_figure_size(self, new_size:tuple):
        self.figure_size = new_size
        self.calculate_pane_size()    

    def set_h_resolution(self, new_h_res:int):
        self.h_res = new_h_res
        self.calculate_pane_size()    


###### --- Functions --- #####

def create_dyn_sys(z, tf_func):
#   generates a second order transfer function of a linear mass spring damper system of unit mass and spring constant, given damping coefficient
#
#   mass spring damper TF: H(s) = 1 / (m*s^2 + b*s + k)
#   w_n = sqrt(k/m) 
#   z   = b / (2*sqrt(km))
#
#   define m as 1, w_n as 1, z as input
#   m = 1
#   k = m * w_n^2 = 1
#   b = 2 * z sqrt(w_n * m) = 2 * z
#
#   parameters: 
#   z[in]   - varying parameter, float
#   tf_func - custom tf function, must be function that accepts a float and returns 2 lists of floats [Num], [Den]
    Num, Den = tf_func(z)
    tf_msd = signal.TransferFunction(Num, Den)
    return tf_msd

def create_time_phase_damp_linspaces(h_res:int, v_res:int, num_oscillations:int, freq_limits:tuple, damping_coeff_limits:tuple):
    # creates sample linspaces for system responses
    # h_res[in]                = total horizontal resolution, int, used for sample time and freq linspace index lengths
    # v_res[in]                = total vertical resolution, int, used for damping coefficient linspace index length
    # num_oscillations[in]     = number of characteristic oscillations in impulse response, sets end time of impulse response 
    # freq_limits[in]
    # damping_coeff_limits[in] = damping coefficient limits, sets start and end of damping coefficients
    t_lin     = np.linspace(0, 2 * np.pi * num_oscillations , h_res)
    w_lin     = np.linspace(freq_limits[0], freq_limits[1], h_res)
    z_lin     = np.linspace(damping_coeff_limits[0],damping_coeff_limits[1], v_res)
    return t_lin, w_lin, z_lin

def generate_mag_phase_imp_arrays(tf, t_lin, w_lin, zeroed_start = True):
    _, mag, phase = signal.bode(tf, w_lin)
    _, imp        = signal.impulse2(tf, X0 = None, T = t_lin)
#   _, imp        = signal.step2(tf, X0 = None, T = t_lin)
    if(zeroed_start == True ):
        mag   = np.array([mag   -   mag[0]])
        phase = np.array([phase - phase[0]])
        imp   = np.array([imp   -   imp[0]])
    else:
        mag   = np.array([mag])
        phase = np.array([phase])
        imp   = np.array([imp])
    return mag, phase, imp

def plot_sos_heatmap(fig, mag_set, phase_set, imp_set, art_def):

    bottom_0    = round(art_def.tb_spacing,    3)
    bottom_1    = round(bottom_0 + art_def.pane_height + art_def.pane_spacing,  3) 
    bottom_2    = round(bottom_1 + art_def.pane_height + art_def.pane_spacing,  3) 
    bottom      = [bottom_2, bottom_1, bottom_0]
    c_scale     = [np.max(np.abs(mag_set)), np.max(np.abs(phase_set + 90)), np.max(np.abs(imp_set))]
    plot_set    = [mag_set, phase_set, imp_set]

    for i in range(3):
        fig.add_axes([0, bottom[i], 1, art_def.pane_height])
        if(art_def.zero_centering==False):
            plt.imshow(plot_set[i], cmap = art_def.color_map, interpolation = art_def.interp_type)
        elif(art_def.zero_centering==True):
            plt.imshow(plot_set[i], cmap = art_def.color_map, interpolation = art_def.interp_type, vmin = -c_scale[i], vmax = c_scale[i])  
        plt.axis('off')

    return

def plot_line_graph_output(out_set, x_lin, idx):
   x_array   = np.array([x_lin])
   plt.plot(x_array.T, out_set[idx])
   plt.show()

def plot_all_line_graph(mag_set, phase_set, imp_set, w_lin, t_lin, loc):
    t_array   = np.array([t_lin])
    w_array   = np.array([w_lin])
    idx       = int(len(mag_set)*loc) 
    if(idx>=len(mag_set)):
        idx = len(mag_set)-1
    plt.subplot(3,1,1)
    plt.plot(w_lin.T, mag_set[idx])
    plt.subplot(3,1,2)
    plt.plot(w_lin.T, phase_set[idx])
    plt.subplot(3,1,3)
    plt.plot(t_lin.T, imp_set[idx])
    plt.show()

##### --- custom tf functions --- #####
def unit_SOS_tf(z):
    m    = 1
    k    = 1
    b    = 2 * z
    Num  = [1]
    Den  = [m, b, k]
    return Num, Den

def unit_lpf(z):
    Num = [z]
    Den = [1, z]
    return Num, Den

def unit_hpf(z):
    Num = [1, 0.0001]
    Den = [1, 1/z]
    return Num, Den

def unit_bpf(z):
    Num = [z, 0.0001]
    Den = [1, 2*z, 1]
    return Num, Den    

def custom_tf_1(z):
    Num = [z, 1]
    Den = [1, z**2, 2]
    return Num, Den

 
##### --- Parameters --- #####


#unit_SOS

tf_type              = unit_SOS_tf
num_oscillations     = 3
freq_limits          = (0, 2.9)
damping_coeff_limits = (0.08, 1)

art_def = art_definitions()
art_def.color_map = 'RdBu'
art_def.interp_type = 'bicubic'
art_def.set_lr_spacing(0.01)
art_def.set_tb_spacing(0.01)
art_def.set_figure_size((15,10))
art_def.set_pane_spacing(0.01)
art_def.set_h_resolution(1000)


##### --- Maths --- #####

t_lin, w_lin, z_lin  = create_time_phase_damp_linspaces(art_def.h_res, art_def.v_res, num_oscillations, freq_limits, damping_coeff_limits) 

for idx, z in enumerate(z_lin):
    tf_msd          = create_dyn_sys(z, tf_type)
    mag, phase, imp = generate_mag_phase_imp_arrays(tf_msd, t_lin, w_lin, zeroed_start = True) 
    if (idx == 0):
        mag_set   = mag
        phase_set = phase
        imp_set   = imp 
    else:
        mag_set   = np.append(mag_set,   mag,   axis=0) 
        phase_set = np.append(phase_set, phase, axis=0) 
        imp_set   = np.append(imp_set,   imp,   axis=0) 

##### --- Plotting --- #####

art_fig = plt.figure(figsize=art_def.figure_size)

plot_sos_heatmap(art_fig, mag_set, phase_set, imp_set, art_def)
# plot_all_line_graph(mag_set, phase_set, imp_set, w_lin, t_lin, 1)
# plot_line_graph_output(mag_set, w_lin, 0)
# plot_line_graph_output(phase_set, w_lin, 0)
# plot_line_graph_output(imp_set, t_lin, 0)
plt.show()

# fig = plt.figure()
# # create all the art subplots
# ax1_art = fig.add_subplot()
# ax1.plot OR gen_art_plots(fig)
# art_axs = [ax1]
# ax2_stuff = fig.add_subplot
# ax2.plot 

# ...
# ok get some information about the current number of subplots
# make a second column of subplots
# move ax2 to the second column of subplots on the figures #idk if this is doable 
# fix all axes aspect ratios 
# if needed maybe fix the canvas to fit everything ("tight?")


# # A new set of data
# time = np.linspace(0, 10, 1000)
# height = np.sin(time)
# weight = time*0.3 + 2
# distribution = np.random.normal(0, 1, len(time))
# # Setting up the plot surface
# fig = plt.figure(figsize=(10, 5))
# gs = GridSpec(nrows=2, ncols=2)
# # First axes
# ax0 = fig.add_subplot(gs[0, 0])
# ax0.plot(time, height)
# # Second axes
# ax1 = fig.add_subplot(gs[1, 0])
# ax1.plot(time, weight)
# # Third axes
# ax2 = fig.add_subplot(gs[:, 1])
# ax2.hist(distribution)
# fig.savefig('figures/gridspec.png')
# plt.show()