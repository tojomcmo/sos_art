from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import impulse as impulseResponse
from scipy.signal import bode as bode



# second order system transfer function
# w_n - natural undamped frequency
# z   - damping coefficient
# w_d  = w_n * sqrt(1 - z^2) 
# H(s) = (w_n^2) / (s^2 + 2*z*w_n*s + w_n^2)

# spring mass damper: H(s) = 1/ (s^2 + (b/m)*s + (k/m)
# w_n = sqrt(k/m)
# z   = (b/m) / (2*sqrt(k/m))

# define m as m, w_n as w_n, z as z
# k = m * w_n^2
# b = 2 * z * w_n * m


def create_dyn_sys(z):

    k   = 1
    b   = 2 * z
    Num    = [1]
    Den    = [1, b, k]
    tf_msd = signal.TransferFunction(Num, Den)

    return tf_msd

def create_time_phase_damp_linspaces(h_res, v_res, num_oscillations, damping_coeff_limits):
    t_lin     = np.linspace(0, 2 * np.pi * num_oscillations , h_res)
    w_lin     = np.linspace(0, 2, h_res)
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
    plt.imshow(mag_set, cmap = color_map, interpolation = interp)
    plt.axis('off')
    plt.subplot(3,1,2)
    plt.imshow(phase_set, cmap = color_map, interpolation = interp)
    plt.axis('off')
    plt.subplot(3,1,3)
    if(zero_centered==False):
        plt.imshow(imp_set, cmap = color_map, interpolation = interp)
    elif(zero_centered==True):
        c_scale = np.max(imp_set)
        plt.imshow(imp_set, cmap = color_map, interpolation = interp, vmin = -c_scale, vmax = c_scale)  
    else:
        Exception(ValueError, 'incorrect zero_center type, ust be True or False')      
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

h_res                = 1000
v_res                = 250
num_oscillations     = 1.45
damping_coeff_limits = (0.09, 1)
color_map            = 'RdBu'
interp               = 'bicubic'
zero_centered_imp    = False

t_lin, w_lin, z_lin  = create_time_phase_damp_linspaces(h_res, v_res, num_oscillations, damping_coeff_limits) 

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

plot_sos_heatmap(mag_set, phase_set, imp_set, color_map, interp, zero_centered = zero_centered_imp)
# plot_all_line_graph(mag_set, phase_set, imp_set, w_lin, t_lin, 0)
# plot_line_graph_output(mag_set, w_lin, 0)
# plot_line_graph_output(phase_set, w_lin, 0)
# plot_line_graph_output(imp_set, t_lin, 0)
