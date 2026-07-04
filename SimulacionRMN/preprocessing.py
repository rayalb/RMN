"""
preprocessing.py
----------------

Core preprocessing utilities for NMR spectra.

This module operates on the Spectrum class and provides:
    * Baseline correction using airPLS.
    * ppm cropping.
    * Internal-standard normalization
    * Binning / resolution reduction.
    * Full prepocessing pipelines.
    
It can be work with:
    * Individual spectra
    * Simulated mixtures
    * dataset generation pipelines
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

from .spectrum import Spectrum


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing pipelines.
    """

    ppm_is_nominal: float = 0.0
    windows_is: float = 0.2

    ppm_min: float = 0.0
    ppm_max: float = 10.0

    lambda_: float = 100.0
    porder: int = 1
    itermax: int = 15

    bin_factor: int = 1

    normalize: bool = True


# Binning

def apply_binning(spectrum : Spectrum, bin_factor : int = 1) -> Spectrum:
    """
    Reduces spectrum resolution by averaging every bin_factor consecutive points.
    Useful for reducing memory footprint while preserving peak shapes.

    Parameters
    ----------
    spectrum  : Spectrum, Input spectrum. 
    bin_factor : int
        Number of points to average.
         Default 1 (no binning). 

    Returns
    -------
    Spectrum, binned spectrum.
    """
    if bin_factor <= 1:
        return spectrum.copy()
    
    intensity = spectrum.intensity
    ppm = spectrum.ppm

    n = len(intensity)
    n_trim = (n//bin_factor)*bin_factor

    intensity_binned = intensity[:n_trim].reshape(-1, bin_factor).mean(axis = 1).astype(np.float32)
    ppm_binned = ppm[:n_trim].reshape(-1, bin_factor).mean(axis = 1)

    return Spectrum(ppm = ppm_binned, intensity = intensity_binned, 
                    metadata = dict(spectrum.metadata or {}))


# airPLS (see paper)

def _whittaker_smooth(x : np.ndarray, w : np.ndarray, lambda_ : float, 
                      differences: int = 1) -> np.ndarray:
    """
    Applies Whittaker smoother to signal x with weights w.
    Internal helper for airPLS baseline correction.

    Parameters
    ----------
    x           : np.ndarray (float, shape=(N,))
        Input signal.
    w           : np.ndarray (float, shape=(N,))
        Weight vector.
    lambda_     : float
        Smoothing parameter. Higher = smoother baseline.
    differences : int
        Order of differences. Default: 1.

    Returns
    -------
    np.ndarray (float, shape=(N,))
        Smoothed background estimate.
    """
    
    m = len(x)
    E = eye(m, format = "csc")
    for _ in range(differences):
        E = E[1:] - E[:-1]
    W = diags(w, 0, shape = (m, m), format = 'csc')
    A = csc_matrix(W + (lambda_*E.T @ E))
    B = W @ x
    return spsolve(A, B)

def airPLS(x : np.ndarray, lambda_ : float = 100, porder : int = 1, 
           itermax : int = 15) -> np.ndarray:
    """
    Adaptive iteratively reweighted penalized least squares (airPLS)
    baseline correction, as used in the reference paper.

    Parameters
    ----------
    x       : np.ndarray (float, shape=(N,))
        Input spectrum signal.
    lambda_ : float
        Smoothing parameter. Default: 100.
    porder  : int
        Order of differences for smoothing. Default: 1.
    itermax : int
        Maximum number of iterations. Default: 15.

    Returns
    -------
    np.ndarray (float, shape=(N,))
        Estimated baseline of the input spectrum.
    """ 
    x = np.asarray(x, dtype = float)
    m = x.shape[0]
    w = np.ones(m)
    for ii in range(1, itermax + 1):
        z = _whittaker_smooth(x, w, lambda_, porder)
        d = x - z
        negative = d[d<0]
        dssn = np.abs(negative.sum())
        if dssn < 0.001*(abs(x)).sum(): 
            break 
        if ii == itermax:
            warnings.warn(
                "airPLS: maximum iterations reached - Baseline may by inaccurate.",
                UserWarning, stacklevel = 2)
            
        w[d >= 0] = 0
        if len(negative) > 0:
            w[d < 0] = np.exp(ii*np.abs(negative)/dssn)
            w[0] = np.exp(ii*negative.max()/dssn)
            w[-1] = w[0]
    return z

def estimate_baseline(spectrum: Spectrum, lambda_: float = 100, porder: int =1, 
                      itermax: int = 15) -> np.ndarray:
    """
    Estimate baseline
    """

    return airPLS(spectrum.intensity, lambda_ = lambda_, porder = porder,
                  itermax = itermax)


def baseline_correct(spectrum : Spectrum, lambda_ : float = 100, porder : int = 1,
                     itermax : int = 15) -> Spectrum:
    """
    Applies airPLS baseline correction.
    
    Returns
    -------
        Spectrum: Baseline-corrected spectrum.
    """

    baseline = estimate_baseline(spectrum, lambda_ = lambda_, porder = porder,
                      itermax = itermax)
    corrected = (spectrum.intensity - baseline).astype(np.float)
    metadata = dict(spectrum.metadata or {})
    metadata["baseline_corrected"] = True

    return Spectrum(ppm = spectrum.ppm.copy(), intensity = corrected,
                    metadata = metadata)

# ppm cropping
def crop_ppm_range(spectrum : Spectrum, ppm_min : float = 0.0, 
                   ppm_max : float = 10.0) -> Spectrum:
    """
    Parameters
    ----------
    spectrum  : Spectrum, Input spectrum. 
    ppm_min   : float
        Lower ppm boundary. Default: 0.0.
    ppm_max   : float
        Upper ppm boundary. Default: 10.0.

    Returns
    -------
    Spectrum, cropped spectrum.
    """
    mask = (spectrum.ppm >= ppm_min) & (spectrum.ppm <= ppm_max)
    return Spectrum(ppm = spectrum.ppm[mask], intensity = spectrum.intensity[mask],
                    metadata = dict(spectrum.metadata or {}))

# Internal standard normalization

def normalize_by_internal_standard(spectrum : Spectrum, ppm_is_nominal : float = 0.0,
                                   window : float = 0.2, eps : float = 1e-12) -> Spectrum:
    """
    Normalizes a spectrum by the height of the internal standard (IS) peak.
    Searches for the maximum absolute intensity within a window around
    the nominal IS ppm position.
    If the IS region is not found or the peak is below eps, returns
    the spectrum unchanged with a warning.

    Parameters
    ----------
    spectrum : Spectrum, Input spectrum.
    ppm_is_nominal : float
        Nominal ppm of the internal standard peak. Default: 0.0 (TSP).
    window         : float
        Half-width of the search window around ppm_is_nominal. Default: 0.2.
    eps            : float
        Minimum peak height to avoid division by zero. Default: 1e-12.

    Returns
    -------
    Spectrum with normalize intensities.
    """
    ppm = spectrum.ppm
    intensity = spectrum.intensity

    mask_is = (ppm >= ppm_is_nominal - window) & (ppm <= ppm_is_nominal + window)

    if not np.any(mask_is):
        warnings.warm(f"Internal-standard peak not found in window [{ppm_is_nominal - window:.2f},"
                      f"{ppm_is_nominal + window:.2f}] ppm. Returning unnormalized spectrum",
                      UserWarning, stacklevel = 2)
        return spectrum.copy()
    
    peak_height = np.max(np.abs(intensity[mask_is]))
    if peak_height < eps:
        warnings.warm("Internal-standard peak height is below eps threshold. Returning unnormalized spectrum",
                      UserWarning, stacklevel = 2)
        return spectrum.copy()
    
    normalized = (intensity / peak_height).astype(np.float32)
    metadata = dict(spectrum.metadata or {})
    metadata["normalized"] = True
    return Spectrum(ppm = ppm.copy(), intensity = normalized, metadata = metadata)

# Full preprocessing pipeline

def preprocess_spectrum(spectrum : Spectrum, ppm_is_nominal : float = 0.0, window_is : float = 0.2,
                        ppm_min : float = 0.0, ppm_max : float = 10.0, lambda_ : float = 100,
                        porder : int = 1, itermax : int = 15, bin_factor : int = 1, 
                        normalize : bool = True) -> Spectrum:
    """
    Full preprocessing pipeline for a single NMR spectrum:
        1. airPLS baseline correction.
        2. Baseline subtraction.
        3. Crop to [ppm_min, ppm_max] window.
        4. Normalization by internal standard peak.
        5. Binning (optional) — averages every bin_factor points.

    Parameters
    ----------
    spectrum : Spectrum, raw input spectrum.
    ppm_is_nominal : float
        Nominal ppm of the internal standard. Default: 0.0 (TSP).
    window_is      : float
        Half-width of IS search window. Default: 0.2.
    ppm_min        : float
        Lower ppm crop boundary. Default: 0.0.
    ppm_max        : float
        Upper ppm crop boundary. Default: 10.0.
    lambda_        : float
        airPLS smoothing parameter. Default: 100.
    porder         : int
        airPLS difference order. Default: 1.
    itermax        : int
        airPLS maximum iterations. Default: 15.
    bin_factor     : int
        Binning factor — average every bin_factor points. Default: 1 (no binning).
        Recommended values: 1 (none), 2, 4. Max safe value depends on peak width.
    normalize      : bool
        if True normalize by standard internal.

    Returns
    -------
    Spectrum : Fully preprocessed spectrum
    """

    processed = baseline_correct(spectrum, lambda_ = lambda_, porder = porder,
                                 itermax = itermax)
    processed = crop_ppm_range(processed, ppm_min = ppm_min, ppm_max = ppm_max)

    if normalize:
        processed = normalize_by_internal_standard(processed, ppm_is_nominal = ppm_is_nominal,
                                                   window = window_is)
    
    processed = apply_binning(processed, bin_factor = bin_factor)

    return processed

def preprocess_many(spectra: list[Spectrum], **kwargs) -> list[Spectrum]:
    """
    Apply preprocessing to many spectra
    """
    return [preprocess_spectrum(spec, **kwargs) for spec in spectra]


# Public API

__all__ = [
    "PreprocessingConfig",
    "apply_binning",
    "airPLS",
    "estimate_baseline",
    "baseline_correct",
    "crop_ppm_range",
    "normalize_by_internal_standard",
    "preprocess_spectrum,"
    "preprocess_many"
]