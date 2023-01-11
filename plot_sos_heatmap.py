from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import impulse as impulseResponse
from scipy.signal import bode as bode
import control as control


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

def create_time_phase_damp_linspaces(h_res:int, v_ratio, num_oscillations:int, freq_limits:tuple, damping_coeff_limits:tuple):
    # creates sample linspaces for system responses
    # h_res[in]                = total horizontal resolution, int, used for sample time and freq linspace index lengths
    # v_res[in]                = total vertical resolution, int, used for damping coefficient linspace index length
    # num_oscillations[in]     = number of characteristic oscillations in impulse response, sets end time of impulse response 
    # freq_limits[in]
    # damping_coeff_limits[in] = damping coefficient limits, sets start and end of damping coefficients
    v_res     = int(np.floor(h_res / v_ratio))
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
    left_coor = round((1-art_def["pane_width"])/2,                                                  2)
    bottom_0  = round((1 - (3 * art_def["pane_aspect_ratio"] + 2 * art_def["pane_spacing"]))/2,     2)
    bottom_1  = round(bottom_0 + art_def["pane_aspect_ratio"] + art_def["pane_spacing"],            2) 
    bottom_2  = round(bottom_1 + art_def["pane_aspect_ratio"] + art_def["pane_spacing"],            2) 
    bottom    = [bottom_0, bottom_1, bottom_2]
    c_scale   = [np.max(np.abs(mag_set)), np.max(np.abs(phase_set + 90)), np.max(np.abs(imp_set))]
    plot_set  = [mag_set, phase_set, imp_set]

    ax_0 = fig.add_axes([left_coor, bottom_0, art_def["pane_width"], art_def["pane_aspect_ratio"]])
    if(art_def["zero_centered"]==False):
        ax_0.imshow(mag_set, cmap = art_def["color_map"], interpolation = art_def["interp"])
    elif(art_def["zero_centered"]==True):
        c_scale = np.max(np.abs(mag_set))
        ax_0.imshow(mag_set, cmap = art_def["color_map"], interpolation = art_def["interp"], vmin = -c_scale, vmax = c_scale)  
    ax_0.axis('off')

    ax_1 = fig.add_axes([left_coor, bottom_1, art_def["pane_width"], art_def["pane_aspect_ratio"]])
    if(art_def["zero_centered"]==False):
        ax_1.imshow(phase_set, cmap = art_def["color_map"], interpolation = art_def["interp"])
    elif(art_def["zero_centered"]==True):
        c_scale = np.max(np.abs(phase_set + 90))
        ax_1.imshow(phase_set + 90, cmap = art_def["color_map"], interpolation = art_def["interp"], vmin = -c_scale, vmax = c_scale)  
    ax_1.axis('off')

    ax_2 = fig.add_axes([left_coor, bottom_2, art_def["pane_width"], art_def["pane_aspect_ratio"]])
    if(art_def["zero_centered"]==False):
        ax_2.imshow(imp_set, cmap = art_def["color_map"], interpolation = art_def["interp"])
    elif(art_def["zero_centered"]==True):
        c_scale = np.max(np.abs(imp_set))
        ax_2.imshow(imp_set, cmap = art_def["color_map"], interpolation = art_def["interp"], vmin = -c_scale, vmax = c_scale)       
    ax_2.axis('off')
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

# #unit_SOS
# h_res                = 1000
# v_ratio              = 5
# num_oscillations     = 3
# tf_type              = unit_SOS_tf
# freq_limits          = (0, 2.9)
# damping_coeff_limits = (0.08, 1)
# color_map            = 'RdBu'
# interp               = 'bicubic'
# zero_centered_imp    = False

#unit_SOS
h_res                = 1000
v_ratio              = 5
tf_type              = unit_bpf
num_oscillations     = 1
freq_limits          = (0, 5)
damping_coeff_limits = (2, 2)
color_map            = 'RdBu'
interp               = 'bicubic'
zero_centered_imp    = False
art_def              = {
                        "color_map"         : 'RdBu',
                        "interp"            : 'bicubic',
                        "pane_width"        : 0.9,
                        "pane_aspect_ratio" : 1/v_ratio,
                        "pane_spacing"      : 0.3,
                        "zero_centered"     : False
                       } 


##### --- Maths --- #####

t_lin, w_lin, z_lin  = create_time_phase_damp_linspaces(h_res, v_ratio, num_oscillations, freq_limits, damping_coeff_limits) 

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

art_fig = plt.figure()

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