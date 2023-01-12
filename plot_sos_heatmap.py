from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import impulse as impulseResponse
from scipy.signal import bode as bode
import control as control

class triptic_dyn_heatmap(object):

    def __init__(self, color_map = 'RdBu', interp_type = 'bicubic', pane_spacing = 0.1, tb_spacing = 0.1, lr_spacing = 0.1, zero_centering = False, h_res:int = 500, figure_size = (12,10)):

        self.color_map       = color_map
        self.interp_type     = interp_type
        self.pane_spacing    = pane_spacing
        self.tb_spacing      = tb_spacing
        self.lr_spacing      = lr_spacing
        self.zero_centering  = zero_centering
        self.h_res           = h_res
        self.figure_size     = figure_size

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

    def set_h_res(self, new_h_res:int):
        self.h_res = new_h_res
        self.calculate_pane_size()    

    def create_dyn_sys(self, z, tf_func):
    #   generates a second order transfer function of a linear mass spring damper system of unit mass and spring constant, given damping coefficient
    #   parameters: 
    #   z[in]   - varying parameter, float
    #   tf_func - custom tf function, must be function that accepts a float and returns 2 lists of floats [Num], [Den]
        Num, Den = tf_func(z)
        tf_instance = signal.TransferFunction(Num, Den)
        return tf_instance   

    def plot_triptic_heatmap(self):

        self.triptic_heatmap = plt.figure(figsize=self.figure_size)
        bottom_0    = round(self.tb_spacing,    3)
        bottom_1    = round(bottom_0 + self.pane_height + self.pane_spacing,  3) 
        bottom_2    = round(bottom_1 + self.pane_height + self.pane_spacing,  3) 
        bottom      = [bottom_2, bottom_1, bottom_0]
        c_scale     = [np.max(np.abs(self.plot_set[0])), np.max(np.abs(self.plot_set[1])), np.max(np.abs(self.plot_set[2]))]
        for i in range(3):
            self.triptic_heatmap.add_axes([0, bottom[i], 1, self.pane_height])
            if(self.zero_centering==False):
                plt.imshow(self.plot_set[i], cmap = self.color_map, interpolation = self.interp_type)
            elif(self.zero_centering==True):
                plt.imshow(self.plot_set[i], cmap = self.color_map, interpolation = self.interp_type, vmin = -c_scale[i], vmax = c_scale[i])  
            plt.axis('off')

        return    

class SOS_triptic_dyn_heatmap(triptic_dyn_heatmap):

    def __init__(self, num_oscillations = 1.45, freq_limits = (0, 2.9), damping_coeff_limits = (0.08, 1), temporal_type = "impulse", zeroed_start = True):
        super().__init__()
        self.num_oscillations     = num_oscillations
        self.freq_limits          = freq_limits
        self.damping_coeff_limits = damping_coeff_limits
        self.temporal_type        = temporal_type
        self.zeroed_start         = zeroed_start

        return

    def create_time_phase_damp_linspaces(self):
        # creates sample linspaces for system responses
        # h_res[in]                = total horizontal resolution, int, used for sample time and freq linspace index lengths
        # v_res[in]                = total vertical resolution, int, used for damping coefficient linspace index length
        # num_oscillations[in]     = number of characteristic oscillations in impulse response, sets end time of impulse response 
        # freq_limits[in]
        # damping_coeff_limits[in] = damping coefficient limits, sets start and end of damping coefficients
        self.t_lin     = np.linspace(0, 2 * np.pi * self.num_oscillations , self.h_res)
        self.w_lin     = np.linspace(self.freq_limits[0], self.freq_limits[1], self.h_res)
        self.z_lin     = np.linspace(self.damping_coeff_limits[0],self.damping_coeff_limits[1], self.v_res)
        return 

    def unit_SOS_tf(self, z):
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
        return Num, Den   

    def generate_mag_phase_imp_arrays(self, tf):
        _, mag, phase = signal.bode(tf, self.w_lin)
        _, imp        = signal.impulse2(tf, X0 = None, T = self.t_lin)
    #   _, imp        = signal.step2(tf, X0 = None, T = t_lin)
        if(self.zeroed_start == True ):
            mag   = np.array([mag   -   mag[0]])
            phase = np.array([phase - phase[0]])
            imp   = np.array([imp   -   imp[0]])
        else:
            mag   = np.array([mag])
            phase = np.array([phase])
            imp   = np.array([imp])    
        return    mag, phase, imp

    def sweep_heatmap_arrays(self):

        self.create_time_phase_damp_linspaces()
        for idx, z in enumerate(self.z_lin):
            tf_instance     = self.create_dyn_sys(z, self.unit_SOS_tf)
            mag, phase, imp = self.generate_mag_phase_imp_arrays(tf_instance) 
            if (idx == 0):
                mag_set   = mag
                phase_set = phase 
                imp_set   = imp 
            else:
                mag_set   = np.append(mag_set,   mag,   axis=0) 
                phase_set = np.append(phase_set, phase, axis=0) 
                imp_set   = np.append(imp_set,   imp,   axis=0) 
        self.plot_set = [mag_set, phase_set+90, imp_set]   
        return

    def plot_all_line_graph(self, loc):
        idx       = int(len(self.plot_set[0])*loc) 
        if(idx>=len(self.plot_set[0])):
            idx = len(self.plot_set[0])-1
        plt.subplot(3,1,1)
        plt.plot(self.w_lin.T, self.plot_set[0][idx])
        plt.subplot(3,1,2)
        plt.plot(self.w_lin.T, self.plot_set[1][idx])
        plt.subplot(3,1,3)
        plt.plot(self.t_lin.T, self.plot_set[2][idx])
        plt.show()



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

SOSart = SOS_triptic_dyn_heatmap()
SOSart.set_figure_size((12,10))
SOSart.set_lr_spacing(0.05)
SOSart.set_tb_spacing(0.1)
SOSart.set_pane_spacing(0.1)
SOSart.set_h_res(50)
SOSart.interp_type = 'Bicubic'

SOSart.sweep_heatmap_arrays()
SOSart.plot_triptic_heatmap()
plt.show()
