# sos_art
Dynamic Art Heatmap Generation
This repository contains a Python script and associated module for generating dynamic heatmap art using second-order system (SOS) dynamics. The main script configures and plots the heatmap, while the module provides the underlying functions and classes.

Install

 - this project uses Poetry to manage install
 - recommended to create and activate a .venv, and install via poetry to .venv

 Main Script
The main script (main.py) is responsible for setting up the parameters, generating the heatmap arrays, and plotting the heatmap art.

Parameters
h_res: Horizontal resolution of the heatmap.
shape_def: Defines the shape based on pane_size or fig_size.
shape: Shape of the heatmap.
discrete_cmap: Whether to use a discrete colormap.
num_discrete_cmap: Number of discrete colors in the colormap.
interp_type: Interpolation type for the heatmap.
Usage
Import necessary modules and functions.
Set the parameters for the heatmap.
Generate the heatmap arrays.
Plot the heatmap art.
Display the plot.

Function Module
The module (src/triptych_dyn_art_funcs.py) contains classes and functions for generating and plotting the dynamic heatmap art.

Classes
 - triptych_dyn_heatmap: Base class for generating triptych dynamic heatmaps.
   - calculate_pane_size: Calculates pane size based on figure size.
   - calculate_fig_size: Calculates figure size based on pane size.
   - create_dyn_sys: Creates a dynamic system using a transfer function.
   - discretize_plot_set: Discretizes the plot set for discrete colormap.
   - plot_triptic_heatmap: Plots the triptych heatmap.
 - SOS_triptych_dyn_heatmap: Derived class for generating SOS triptych dynamic heatmaps.
   - create_time_phase_damp_linspaces: Creates sample linspaces for system responses.
   - generate_mag_phase_imp_arrays: Generates magnitude, phase, and impulse arrays.
   - sweep_heatmap_arrays: Sweeps through heatmap arrays to generate the data set.
   - lot_all_line_graph: Plots all line graphs.
Functions
 - unit_SOS_tf(z): Generates a unit SOS transfer function.
 - two_mass_f1_to_m1_tf(z): Generates a transfer function for a two-mass system.
 - unit_lpf(z): Generates a unit low-pass filter transfer function.
 - unit_hpf(z): Generates a unit high-pass filter transfer function.
 - unit_bpf(z): Generates a unit band-pass filter transfer function.
 - custom_tf_1(z): Generates a custom transfer function.
Usage
Import the module and use the classes and functions to generate dynamic heatmap art.