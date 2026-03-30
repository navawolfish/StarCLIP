# Author: Nava Wolfish
# University of Toronto
# nava.wolfish19@gmail.com
# 
# 
# This file contains the Spectra class and related functions for processing and analyzing observed spectra with synthetic templates. It includes methods for continuum fitting, anomaly detection, and visualization of spectral data, particularly for handling detector gaps in NIR spectrographs. 
# 
# This class is used in the JWST spectral reduction pipeline, largely in the 1d_to_starclip_reduction notebook, but also in the rest_frame_correct notebook. 

#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from matplotlib import cm

#matplotlib params
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

#quadratic function for fitting
def quadratic(x, a, b, c):
    """
    Compute a quadratic polynomial: f(x) = ax² + bx + c.

    Parameters
    ----------
    x : np.ndarray
        Input values at which to evaluate the quadratic.
    a : float
        Coefficient of the quadratic term (x²).
    b : float
        Coefficient of the linear term (x).
    c : float
        Constant term (y-intercept when x=0).

    Returns
    -------
    np.ndarray
        Quadratic polynomial evaluated at each point in x.
    """
    return a * x**2 + b * x + c


#main class definition 
class Spectra:
    """
    A class for processing and analyzing observed spectra with synthetic templates.

    This class provides tools for continuum fitting, anomaly detection, and
    visualization of spectral data, particularly for handling detector gaps
    in NIR spectrographs.

    Parameters
    ----------
    wl : np.ndarray
        Wavelength array of the observed spectrum (in Angstroms).
    spec : np.ndarray
        Flux values of the observed spectrum.
    unc : np.ndarray
        Flux uncertainties for the observed spectrum.
    synth_wl : np.ndarray
        Wavelength array of the synthetic/template spectrum.
    synth_spec : np.ndarray
        Flux values of the synthetic/template spectrum.

    Attributes
    ----------
    wavelength : np.ndarray
        Observed spectrum wavelength array.
    spectrum : np.ndarray
        Observed spectrum flux values.
    uncertainty : np.ndarray
        Observed spectrum flux uncertainties.
    synth_wavelength : np.ndarray
        Synthetic spectrum wavelength array.
    synth_spectrum : np.ndarray
        Synthetic spectrum flux values.
    rm : np.ndarray or None
        Rolling median continuum estimate.
    quadratic_ctm : np.ndarray or None
        Quadratic continuum fit across full spectrum.
    quadratic_params : dict or None
        Fitted quadratic coefficients for left and right detector regions.
    """
    def __init__(self, wl, spec, unc, synth_wl, synth_spec):
        """ Initialize the Spectra class with observed and synthetic spectrum data.
        Parameters
        ----------
        wl : np.ndarray
            Wavelength array of the observed spectrum (in Angstroms).
        spec : np.ndarray
            Flux values of the observed spectrum.
        unc : np.ndarray
            Flux uncertainties for the observed spectrum.
        synth_wl : np.ndarray
            Wavelength array of the synthetic/template spectrum.
        synth_spec : np.ndarray
            Flux values of the synthetic/template spectrum.
        """
        self.wavelength = wl
        self.spectrum = spec
        self.uncertainty = unc
        self.synth_wavelength = synth_wl
        self.synth_spectrum = synth_spec

        #set these to None for now, will be filled in by methods
        self.rm = None
        self.quadratic_ctm = None

        self.quadratic_left = None
        self.quadratic_right = None

        self.quadratic_params = None

    def rolling_median(self, data, window_size=50):
        """
        Compute a rolling median filter as a continuum estimate.

        Applies a median filter to smooth the spectrum and estimate the
        underlying continuum level. The result is stored in `self.rm`.

        Parameters
        ----------
        data : np.ndarray
            Input flux array to filter.
        window_size : int, optional
            Width of the median filter window in pixels. Default is 50.

        Returns
        -------
        np.ndarray
            Rolling median filtered data.
        """
        rm = median_filter(data, size=window_size, mode='nearest')
        self.rm = rm
        return rm
    
    def anomaly_mask(self, data, sigma_threshold):
        """
        Identify outlier pixels that deviate significantly from the continuum.

        Computes residuals between the data and rolling median continuum,
        then flags pixels exceeding the specified sigma threshold.

        Parameters
        ----------
        data : np.ndarray
            Input flux array to check for anomalies.
        sigma_threshold : float
            Number of standard deviations above which a pixel is flagged
            as an anomaly.

        Returns
        -------
        np.ndarray
            Boolean mask where True indicates anomalous pixels.

        Raises
        ------
        ValueError
            If `rolling_median()` has not been called first.
        """
        if self.rm is None:
            raise ValueError("Rolling median continuum not calculated. Please run rolling_median() first.")
        continuum = self.rm
        residuals = data - continuum
        std_dev = np.std(residuals)
        anomaly_mask = np.abs(residuals) > (sigma_threshold * std_dev)
        return anomaly_mask

    def detector_gap(self):
        """
        Find the index where the detector gap occurs in the wavelength array.

        Detects the largest discontinuity in the wavelength grid, which
        corresponds to the gap between detector chips in NIR spectrographs.

        Returns
        -------
        int
            Index marking the start of the right detector region
            (first pixel after the gap).
        """
        detector_gap_idx = np.argmax(np.diff(self.wavelength)) + 1 #+1 to get start of right side
        return detector_gap_idx
    
    
    def quadratic_fit(self, mask = False):
        """
        Fit quadratic continuum models to both detector regions.

        Performs separate quadratic fits to the left and right portions
        of the spectrum (split at the detector gap) to model the continuum.
        Results are stored in `self.quadratic_ctm` and `self.quadratic_params`.

        Parameters
        ----------
        mask : bool, optional
            If True, exclude 3-sigma outliers (identified via `anomaly_mask`)
            from the fit. Requires `rolling_median()` to be called first.
            Default is False.

        Returns
        -------
        dict
            Dictionary with keys 'Left' and 'Right', each containing the
            fitted quadratic coefficients [a, b, c].

        Raises
        ------
        ValueError
            If `mask=True` but `rolling_median()` has not been called.
        """
        if mask:
            if self.rm is None:
                raise ValueError("Rolling median continuum not calculated. Please run rolling_median() first.")
            anomaly_mask = self.anomaly_mask(self.spectrum, sigma_threshold=3)
            # Apply mask to data
            wl = self.wavelength[~anomaly_mask]
            spec = self.spectrum[~anomaly_mask]
            unc = self.uncertainty[~anomaly_mask]

        else:
            wl = self.wavelength
            spec = self.spectrum
            unc = self.uncertainty

        detector_gap_idx = self.detector_gap()

        #split into two regions
        real_grid_left = wl[:detector_gap_idx] # Left side wavelengths
        real_spec_left = spec[:detector_gap_idx] # Left side spectra
        spec_unc_left = unc[:detector_gap_idx] # Left side

        real_grid_right = wl[detector_gap_idx:] # Right side wavelengths
        real_spec_right = spec[detector_gap_idx:] # Right side spectra
        spec_unc_right = unc[detector_gap_idx:] # Right side uncertainties

        # Fit quadratic to left side
        popt_left, pcov_left = curve_fit(quadratic, real_grid_left, real_spec_left, sigma=spec_unc_left, absolute_sigma=True)
        self.quadratic_left = quadratic(self.wavelength[:detector_gap_idx], *popt_left)
        # Fit quadratic to right side
        popt_right, pcov_right = curve_fit(quadratic, real_grid_right, real_spec_right, sigma=spec_unc_right, absolute_sigma=True)
        self.quadratic_right = quadratic(self.wavelength[detector_gap_idx:], *popt_right)

        # Combine both sides
        self.quadratic_ctm = np.concatenate((self.quadratic_left, self.quadratic_right))
        self.quadratic_params = {'Left': popt_left, 'Right': popt_right}
        return self.quadratic_params
    
    def plot(self):
        """
        Visualize the observed and synthetic spectra with continuum fits.

        Creates a publication-ready figure showing:
        - Observed spectrum (blue)
        - Synthetic/template spectrum (orange)
        - Rolling median continuum (red, if computed)
        - Quadratic continuum fit (green, if computed)
        """
        fig, ax = plt.subplots( figsize = (20, 3), sharex=True)

        ax.plot(self.wavelength, self.spectrum, color = cs[0], label='Observed Spectrum')
        ax.plot(self.synth_wavelength, self.synth_spectrum, color = cs[2], label='Synthetic Spectrum')

        if self.rm is not None:
            ax.plot(self.wavelength, self.rm, color = 'red', label = 'Rolling Median Continuum', linewidth=2)
        
        if self.quadratic_ctm is not None:
            ax.plot(self.wavelength, self.quadratic_ctm, color = 'green', label = 'Quadratic Fit', linewidth=2)

        ax.set_xlim([min(self.wavelength), max(self.wavelength)])
        ax.set_title(r'\textbf{Observed vs Synthetic Spectrum}')
        ax.set_xlabel(r'\textbf{Wavelength}')
        ax.set_ylabel(r'\textbf{Flux}')
        ax.legend()
        plt.tight_layout()
        plt.show()
