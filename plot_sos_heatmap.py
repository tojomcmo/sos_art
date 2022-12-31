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


def create_dyn_sys(w_n, z):

    k   = w_n**2
    b   = 2 * z * w_n
    Num    = [1]
    Den    = [1, b, k]
    tf_msd = signal.TransferFunction(Num, Den)

    return tf_msd

# def create_time_phase_damp_arrays(h_res, a_ratio, ):

def plot_sos_heatmap(mag_set, phase_set, imp_set, color_map, interp):
    plt.subplot(3,1,1)
    plt.imshow(mag_set, cmap = color_map, interpolation = interp)
    plt.axis('off')
    plt.subplot(3,1,2)
    plt.imshow(phase_set, cmap = color_map, interpolation = interp)
    plt.axis('off')
    plt.subplot(3,1,3)
    plt.imshow(imp_set, cmap = color_map, interpolation = interp)
    plt.axis('off')
    plt.show()



iter = 200
time_samples =1000
freq_samples = 1000

w_n = 1
t_lin     = np.linspace(0, np.pi*3/w_n, time_samples)
w_lin     = np.linspace(0, w_n*2 , freq_samples)
t_array   = np.array([t_lin])
w_array   = np.array([w_lin])

z_vec     = np.linspace(0.09,1, num = iter)

for idx, z in enumerate(z_vec):
    tf_msd        = create_dyn_sys(w_n, z)
    _, mag, phase = signal.bode(tf_msd, w_lin)
    _, imp        = signal.impulse2(tf_msd, X0 = None, T = t_lin)
    mag   = np.array([mag-mag[0]])
    phase = np.array([phase])
    imp   = np.array([imp])
    if (idx == 0):
        mag_set   = mag
        phase_set = phase
        imp_set   = imp 
    else:
        mag_set   = np.append(mag_set,   mag,   axis=0) 
        phase_set = np.append(phase_set, phase, axis=0) 
        imp_set   = np.append(imp_set,   imp,   axis=0) 

color_map = 'RdBu'
interp = 'bicubic'

plot_sos_heatmap(mag_set, phase_set, imp_set, color_map, interp)
