from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import impulse as impulseResponse
from scipy.signal import bode as bode



###### --- Functions --- #####

def create_dyn_sys(z):
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
    m    = 1
    k    = 1
    b    = 2 * z
    Num  = [1]
    Den  = [m, b, k]
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

def generate_mag_phase_imp_arrays(tf, t_lin, w_lin):
    _, mag, phase = signal.bode(tf, w_lin)
    _, imp        = signal.impulse2(tf, X0 = None, T = t_lin)
    mag   = np.array([mag   -   mag[0]])
    phase = np.array([phase - phase[0]])
    imp   = np.array([imp   -   imp[0]])
    return mag, phase, imp

def plot_sos_heatmap(mag_set, phase_set, imp_set, color_map, interp, zero_centered = False):
    plt.subplot(3,1,1)
    if(zero_centered==False):
        plt.imshow(mag_set, cmap = color_map, interpolation = interp)
    elif(zero_centered==True):
        c_scale = np.max(np.abs(mag_set))
        plt.imshow(mag_set, cmap = color_map, interpolation = interp, vmin = -c_scale, vmax = c_scale)  
    plt.axis('off')
    plt.subplot(3,1,2)
    if(zero_centered==False):
        plt.imshow(phase_set, cmap = color_map, interpolation = interp)
    elif(zero_centered==True):
        c_scale = np.max(np.abs(phase_set + 90))
        plt.imshow(phase_set + 90, cmap = color_map, interpolation = interp, vmin = -c_scale, vmax = c_scale)  
    plt.axis('off')
    plt.subplot(3,1,3)
    if(zero_centered==False):
        plt.imshow(imp_set, cmap = color_map, interpolation = interp)
    elif(zero_centered==True):
        c_scale = np.max(np.abs(imp_set))
        plt.imshow(imp_set, cmap = color_map, interpolation = interp, vmin = -c_scale, vmax = c_scale)       
    plt.axis('off')
    plt.show()

def plot_line_graph_output(out_set, x_lin, idx):
   x_array   = np.array([x_lin])
   plt.plot(x_array.T, out_set[idx])
   plt.show()

def plot_all_line_graph(mag_set, phase_set, imp_set, w_lin, t_lin, idx):
    t_array   = np.array([t_lin])
    w_array   = np.array([w_lin])
    plt.subplot(3,1,1)
    plt.plot(w_lin.T, mag_set[idx])
    plt.subplot(3,1,2)
    plt.plot(w_lin.T, phase_set[idx])
    plt.subplot(3,1,3)
    plt.plot(t_lin.T, imp_set[idx])
    plt.show()


##### --- Parameters --- #####

h_res                = 1000
v_ratio              = 5
num_oscillations     = 3
freq_limits          = (0, 2.9)
damping_coeff_limits = (0.08, 1)
color_map            = 'RdBu'
interp               = 'bicubic'
zero_centered_imp    = False

##### --- Maths --- #####

t_lin, w_lin, z_lin  = create_time_phase_damp_linspaces(h_res, v_ratio, num_oscillations, freq_limits, damping_coeff_limits) 

for idx, z in enumerate(z_lin):
    tf_msd          = create_dyn_sys(z)
    mag, phase, imp = generate_mag_phase_imp_arrays(tf_msd, t_lin, w_lin) 
    if (idx == 0):
        mag_set   = mag
        phase_set = phase
        imp_set   = imp 
    else:
        mag_set   = np.append(mag_set,   mag,   axis=0) 
        phase_set = np.append(phase_set, phase, axis=0) 
        imp_set   = np.append(imp_set,   imp,   axis=0) 

##### --- Plotting --- #####

plot_sos_heatmap(mag_set, phase_set, imp_set, color_map, interp, zero_centered = zero_centered_imp)
# plot_all_line_graph(mag_set, phase_set, imp_set, w_lin, t_lin, 0)
# plot_line_graph_output(mag_set, w_lin, 0)
# plot_line_graph_output(phase_set, w_lin, 0)
# plot_line_graph_output(imp_set, t_lin, 0)
