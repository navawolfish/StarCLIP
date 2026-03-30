# Author: Nava Wolfish
# University of Toronto
# nava.wolfish19@gmail.com 
#
# 
# This script contains the rest frame correction function, which finds the best Doppler shift to align real and synthetic spectra. It uses a Gaussian fit to the chi-squared values to find the best shift, and includes plotting functionality to visualize the results.
# 
# We source this script in the notebook 1d_to_starclip_reduction.ipynb, and use the rest_frame_correction function within the Spectra class in spectra.py to apply the correction to our spectra.

#%% Imports
import os
from pathlib import Path

# Plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from tqdm import tqdm

# Science
import numpy as  np
from astropy.io import fits
import pandas as pd

# PypeIt
import pypeit
from pypeit.metadata import PypeItMetaData
from pypeit.spectrographs.util import load_spectrograph
from pypeit import spec2dobj, specobjs

# Processing
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

#%% Set plotting style
plt.rcParams.update({
    "figure.figsize": (8, 5),  # Default figure size
    "axes.titlesize": 16,      # Title font size
    "axes.labelsize": 14,      # Axis label font size
    "xtick.labelsize": 12,     # X-tick label font size
    "ytick.labelsize": 12,     # Y-tick label font size
    "legend.fontsize": 12,     # Legend font size
    "lines.linewidth": 2,      # Line width
    "grid.alpha": 0.5,         # Grid transparency
    "grid.linestyle": "--",    # Grid line style
    "axes.grid": True,         # Show grid
    "axes.facecolor": "#F6F5F3", # Axes background color
    "savefig.dpi": 300,        # Default DPI for saving figures
    "text.usetex": True,       # Use LaTeX for text rendering
    "font.family": "serif",    # Use serif fonts
})

cs = ["#335c67","#fff3b0","#e09f3e","#9e2a2b","#540b0e", "#b3b3cc"] # Color palette for plots
# %% helper functions
def doppler_shift(wave, rv):
    """Applies a Doppler shift to the given wavelength array.

    Parameters:
    wave (np.ndarray): The original wavelength array.
    rv (float): The Doppler shift to apply.

    Returns:
    np.ndarray: The Doppler-shifted wavelength array.
    """
    return wave * (1 - rv / 3e5)  # Assuming v is in km/s

def chi_squared(synth, dop_shift, spec_unc):
    """Calculates the chi-squared statistic between synthetic and doppler shifted data.

    Parameters:
    synth (np.ndarray): The synthetic data.
    dop_shift (np.ndarray): The doppler shifted data.
    spec_unc (np.ndarray): The uncertainty in the spectrum.

    Returns:
    float: The chi-squared statistic.
    """
    return abs(np.sum((dop_shift - synth) ** 2 / spec_unc**2))


def wl_cut(wl_grid, spec, wl_lims):
    """Cuts the wavelength grid and spectrum to the specified limits.

    Parameters:
    wl_grid (np.ndarray): The wavelength grid.
    spec (np.ndarray): The spectrum data.
    wl_lims (tuple): The wavelength limits (min, max).

    Returns:
    tuple: The cut wavelength grid and spectrum.
    """
    #if wl_lims is only one tuple, 
    if isinstance(wl_lims, tuple):
        wl_lims = [wl_lims] #make it a list of tuples
        wl_lims = np.array(wl_lims)

    
    cut_mask = np.zeros_like(wl_grid, dtype=bool)
    #keep only if within any of the limits
    for lim in wl_lims:
        cut_mask |= (wl_grid >= lim[0]) & (wl_grid <= lim[1])
    
    # print(cut_mask.shape, wl_grid.shape, spec.shape)
    return wl_grid[cut_mask], spec[cut_mask]

def gaussian(x, amp, mean, stddev, offset):
    """Gaussian function for curve fitting.

    Parameters:
    x (np.ndarray): The input array.
    amp (float): Amplitude of the Gaussian.
    mean (float): Mean of the Gaussian.
    stddev (float): Standard deviation of the Gaussian.

    Returns:
    np.ndarray: The Gaussian function evaluated at x.
    """
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + offset

#%% Steps within main function
def doppler_and_interp(rv, synth_grid, synth_spec, wl_grid, wl_lims, real_spec_cut, spec_unc_cut, returns = 'chi2'):
        """Doppler shifts the synthetic spectrum, interpolates it to the real wavelength grid, and calculates chi-squared.
        Parameters:
        rv (float): The Doppler shift to apply.
        synth_grid (np.ndarray): The wavelength grid of the synthetic spectrum.
        synth_spec (np.ndarray): The synthetic spectrum data.
        wl_grid (np.ndarray): The wavelength grid of the real spectrum.
        wl_lims (tuple): The wavelength limits (min, max).
        real_spec_cut (np.ndarray): The cut real spectrum data.
        spec_unc_cut (np.ndarray): The cut uncertainty in the real spectrum.
        returns (str): What to return, either 'chi2' for chi-squared or other for the shifted spectrum.
        Returns:
        float or np.ndarray: The chi-squared statistic or the doppler shifted synthetic spectrum on the real wavelength grid.
        """
        #step 2a: doppler correct synth spectrum
        synth_grid_dopp = doppler_shift(synth_grid, -rv)

        # step 2b: select synth grid points and spectral points within limits
        synth_grid_dopp, synth_spec_cut = wl_cut(synth_grid_dopp, synth_spec, wl_lims)

        # step 2c: interpolate to real wavelength grid
        wl_grid_clipped = np.clip(wl_grid, synth_grid_dopp.min(), synth_grid_dopp.max()) #so no errors
        interpolator = interp1d(synth_grid_dopp, synth_spec_cut, kind='linear', fill_value=0)
        synth_spec_real_grid = interpolator(wl_grid_clipped)

        # determine chi squared
        chi_sq = chi_squared(synth_spec_real_grid, real_spec_cut, spec_unc_cut)
        if returns == 'chi2':
            return chi_sq
        else:
            return synth_spec_real_grid

def fit_gaussian_to_chi2(rvs, chi2s):
    """Fits a Gaussian to the chi-squared values to find the best Doppler shift. If the fit fails, it falls back to using the minimum chi-squared value and returns None for the fit parameters.

    Parameters:
    rvs (np.ndarray): The array of Doppler shifts.
    chi2s (np.ndarray): The corresponding chi-squared values.

    Returns:
    float: The best Doppler shift value.
    tuple: The fit parameters (amplitude, mean, stddev, offset) or None if the fit fails.
    """
    # Initial guess for Gaussian parameters: amplitude, mean, stddev, offset
    initial_guess = [min(chi2s), rvs[np.argmin(chi2s)], 10, max(chi2s)]

    # Fit Gaussian
    try:
        popt, _ = curve_fit(gaussian, rvs, chi2s, p0=initial_guess)
        best_rv = popt[1]
    except RuntimeError:
        # print("Gaussian fit failed, using minimum chi-squared value.")
        best_rv = rvs[np.argmin(chi2s)]  # Fallback to minimum chi2 if fit fails
        popt = None
    return best_rv, popt

def do_plots(rvs, chi2s, cts_gaus, best_rv, synth_grid, 
             synth_spec, wl_grid_cut, wl_lims, real_spec_cut, spec_unc_cut):
    """
    Plots the chi-squared values against the Doppler shifts and the best fit synthetic spectrum against the real spectrum.
    Parameters:
    rvs (np.ndarray): The array of Doppler shifts.
    chi2s (np.ndarray): The corresponding chi-squared values.
    best_rv (float): The best Doppler shift value.
    synth_grid (np.ndarray): The wavelength grid of the synthetic spectrum.
    synth_spec (np.ndarray): The synthetic spectrum data.
    wl_grid_cut (np.ndarray): The cut wavelength grid of the real spectrum.
    wl_lims (tuple): The wavelength limits (min, max).
    real_spec_cut (np.ndarray): The cut real spectrum data.
    spec_unc_cut (np.ndarray): The cut uncertainty in the real spectrum.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rvs, chi2s, label='Chi-squared', color=cs[4])
    plt.plot(rvs, cts_gaus, label='Gaussian Fit', linestyle='--', color=cs[2])
    plt.axvline(best_rv, color='black', label=f'Best RV: {best_rv:.2f} km/s')
    plt.xlabel(r'\textbf{Radial Velocity} [km/s]')
    plt.ylabel(r'\textbf{Chi-squared}')
    plt.title(r'\textbf{Chi-squared vs Radial Velocity}')
    plt.legend()
    plt.show()

    # Plot best fit
    best_fit_spec = doppler_and_interp(best_rv, synth_grid, synth_spec, wl_grid_cut, wl_lims, real_spec_cut, spec_unc_cut, returns='synth')
    plt.figure(figsize = (20, 6))
    plt.plot(wl_grid_cut, real_spec_cut / np.mean(real_spec_cut), label='Real Spectrum', marker='.', color = cs[0])
    plt.plot(wl_grid_cut, best_fit_spec, label='Best Fit Synthetic Spectrum', marker = '.', color = cs[2])
    plt.xlabel(r'\textbf{Wavelength} [$\AA$]')
    plt.ylabel(r'\textbf{Flux} [arbitrary units]')
    plt.title(r'\textbf{Best Fit Synthetic Spectrum vs Real Spectrum}', )
    plt.legend()
    plt.show()
    return

def gaus_plot(rvs, chi2s, cts_gaus, best_rv):
    """ Plots the chi-squared values against the Doppler shifts and the Gaussian fit, highlighting the best Doppler shift.
    Parameters:
    rvs (np.ndarray): The array of Doppler shifts.
    chi2s (np.ndarray): The corresponding chi-squared values.
    cts_gaus (np.ndarray): The Gaussian fit values for the chi-squared data
    best_rv (float): The best Doppler shift value.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rvs, chi2s, label='Chi-squared', color=cs[4])
    plt.plot(rvs, cts_gaus, label='Gaussian Fit', linestyle='--', color=cs[2])
    plt.axvline(best_rv, color='black', label=f'Best RV: {best_rv:.2f} km/s')
    plt.xlabel(r'\textbf{Radial Velocity} [km/s]')
    plt.ylabel(r'\textbf{Chi-squared}')
    plt.title(r'\textbf{Chi-squared vs Radial Velocity}')
    plt.legend()
    plt.show()
    return 
#%% Main function
def do_plots2(wl_lims, real_grid, real_spec, synth_grid, synth_spec, rv):
    """
    For a list of wavelength limits, create N subplots (N = len(wl_lims)),
    and plot both real and synthetic spectra on each axis according to the appropriate wavelength range.
    Parameters:
    wl_lims (list of tuple): List of wavelength limits (min, max) for each subplot.
    real_grid (np.ndarray): Wavelength grid for the real spectrum.
    real_spec (np.ndarray): Real spectrum data.
    synth_grid (np.ndarray): Wavelength grid for the synthetic spectrum.
    synth_spec (np.ndarray): Synthetic spectrum data.
    """
    
    N = len(wl_lims)
    fig, axes = plt.subplots(N, 1, figsize=(10, 3*N), sharey=False)
    if N == 1:
        axes = [axes]
    for i, lim in enumerate(wl_lims):
        # Cut real and synthetic spectra to the wavelength limits
        wl_real_cut, real_spec_cut = wl_cut(real_grid, real_spec, lim)
        wl_synth_cut, synth_spec_cut = wl_cut(synth_grid, synth_spec, lim)
        ax = axes[i]
        ax.plot(wl_real_cut, real_spec_cut / np.mean(real_spec_cut), label='Real Spectrum', color=cs[0], marker='.')
        ax.plot(wl_synth_cut, synth_spec_cut / np.mean(synth_spec_cut), label='Synthetic Spectrum', color=cs[2], marker='.')
        ax.set_xlabel(r'Wavelength [$\AA$]')
        ax.set_ylabel('Flux (norm.)')
        ax.set_title(f'Wavelength Range: {lim[0]} - {lim[1]}')
        ax.legend()
    plt.tight_layout()
    plt.suptitle(r'\textbf{Best Fit Synthetic Spectrum vs Real Spectrum, rv = {rv}}', y=1.02)
    plt.show()
    return
def rest_frame_correction(real_grid, synth_grid, real_spec, synth_spec, spec_unc, central_rv, wl_lims = None, plot=False, prev_range=20):
    """Finds the best Doppler shift to align real and synthetic spectra.

    Parameters:
    real_grid (np.ndarray): The wavelength grid of the real spectrum.
    synth_grid (np.ndarray): The wavelength grid of the synthetic spectrum.
    real_spec (np.ndarray): The real spectrum data.
    synth_spec (np.ndarray): The synthetic spectrum data.
    spec_unc (np.ndarray): The uncertainty in the real spectrum.
    central_rv (float): The central radial velocity around which to search for the best shift.
    wl_lims (tuple, optional): The wavelength limits (min, max) to consider for the correction. If None, it will use the overlapping range of the real and synthetic spectra.
    plot (bool, optional): Whether to plot the chi-squared values and the best fit spectrum. Default is False.
    prev_range (float, optional): The initial range around central_rv to search for the best shift. This will be increased if the Gaussian fit fails. Default is 20 km/s.

    Returns:
    float: The best Doppler shift value.
    """
    #step 1: cut to wavelength limits
    if wl_lims is None:
        wl_lims = (max(real_grid.min(), synth_grid.min()), min(real_grid.max(), synth_grid.max()))
    
    # cut real spectrum to limits
    wl_grid_cut, real_spec_cut = wl_cut(real_grid, real_spec, wl_lims)

    _, spec_unc_cut = wl_cut(real_grid, spec_unc, wl_lims)

    #step 2: doppler correct synth spectrum
    popt = None
    its = 0
    while popt is None and its < 10: #keep going until gaussian fit works
        prev_range = prev_range + 10
        its += 1 #escape hatch in case of infinite loop, should never happen but just in case
        # Create RV array around central_rv
        rvs = np.linspace(central_rv - prev_range, central_rv + prev_range, 10000)
        chi2s = np.zeros(len(rvs)) #store chi squared values
        for i, rv in enumerate(rvs):
            chi2s[i] = doppler_and_interp(rv, synth_grid, synth_spec, wl_grid_cut, wl_lims, real_spec_cut, spec_unc_cut, returns='chi2')

        #step 3: find best rv
        best_rv, popt = fit_gaussian_to_chi2(rvs, chi2s)

        if its == 10 and popt is None:
            print("Gaussian fit failed after 10 iterations.")
            return None


    if plot: #plot results
        cts_gaus = gaussian(rvs, *popt) # get the Gaussian fit values for plotting
        gaus_plot(rvs, chi2s, cts_gaus, best_rv)
        # Doppler-shift the full synthetic grid for plotting
        synth_grid_dopp = doppler_shift(synth_grid, -best_rv)
        do_plots2(wl_lims, real_grid, real_spec, synth_grid_dopp, synth_spec, best_rv)
    return best_rv
# %%
