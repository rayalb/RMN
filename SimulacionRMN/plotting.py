"""
plotting.py
-----------

Visualization utilities for NMR spectra.

Supports:
* Single spectrum
* Multiple spectra
* Mixtures
* Difference spectra
"""

from __future__ import annotations
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np

from .spectrum import Spectrum, MixtureSpectrum
from .library import SpectrumLibrary

def plot_spectrum(spectrum: Spectrum, ax = None, title: str | None = None,
                  invert_ppm: bool = True, linewidth: float = 1.0,
                  show: bool = True):
    """
    Plot a single spectrum.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize = (10, 4))

    ax.plot(spectrum.ppm, spectrum.intensity, linewidth = linewidth,
            label = spectrum.name)
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")

    if title is not None:
        ax.set_title(title)
    elif spectrum.name:
        ax.set_title(spectrum.name)

    if invert_ppm:
        ax.invert_xaxis()

    if spectrum.name:
        ax.legend()

    if show:
        plt.show()

    return ax

def plot_spectra(spectra: Sequence[Spectrum], labels: Sequence[str] | None = None, 
                 ax = None, title: str | None = None,
                 invert_ppm: bool = True, linewidth: float = 1.0,
                 alpha: float = 0.9, show: bool = True):
    """
    Overlay multiple spectra.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize = (10, 4))

    for ii, spec in enumerate(spectra):
        if labels is not None:
            label = labels[ii]
        elif spec.name:
            label = spec.name

        ax.plot(spec.ppm, spec.intensity, linewidth = linewidth, alpha = alpha,
                label = label)

    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")

    if title:
        ax.set_title(title)

    if invert_ppm:
        ax.invert_xaxis()

    ax.legend()

    if show:
        plt.show()

    return ax

def plot_mixture(mixture: MixtureSpectrum, show_composition: bool = True,
                   ax = None, show: bool = True):
    """
    Plot mixture spectrum.
    """

    title = "Synthetic Mixture"

    if show_composition:
        comps = ", ".join(mixture.composition.keys())
        title += f"\n{comps}"

    return plot_spectrum(mixture, ax = ax, title = title, show = show)

def plot_mixture_components(mixture: MixtureSpectrum, library: SpectrumLibrary | None = None, 
                            show_sum: bool = True, mixture_on_top: bool = True, 
                            vertical_offset: float | None = None,
                            invert_ppm: bool = True, linewidth: float = 1.2,
                            alpha: float = 0.7, show: bool = True):
    """
    plot every component of a mixture, optionally with the sum spectrum.

    Parameters
    ----------
        mixture: MixtureSpectrum, final simalted mixture.
        library: SpectrumLibrary, optional, library containing the component spectra.
        show_sum: bool, optional, whether to show the sum spectrum.
        mixture_on_top: bool, optional, plot the mixture above the individual spectra.
        vertical_offset: float, optional, vertical separation between spectra. 
                        If None, it will be automatically calculated.
    Returns
    -------
        ax: matplotlib axes object.
    """

    fig, ax = plt.subplots(figsize = (10, 6))

    if mixture.components:
        components = mixture.components
    else:
        if library is None:
            raise ValueError(
                "Mixture does not contain component spectra."
                "Provide a SpectrumLibrary or generate the mixture with store_components = True.")
        
        components = {
            name: library.get_raw(name) for name in mixture.composition
            }
    max_height = max(np.max(np.abs(spec.intensity*mixture.composition[name])) 
                     for name, spec in components.items())
    
    if show_sum:
        max_height = max(max_height, np.max(np.abs(mixture.intensity)))

    if vertical_offset is None:
        vertical_offset = 1.3*max_height

    n = len(components)                      

    if show_sum and mixture_on_top:
        baseline = n*vertical_offset
        ax.axhline(baseline, color = "lightgray", linewidth = 1.5, 
                   zorder = 0)
        ax.plot(mixture.ppm, mixture.intensity + baseline, label = "Mixture",
                linewidth = 1.5, color = "black")

    for ii, (name, spec) in enumerate(components.items()): 
        conc = mixture.composition[name]
        baseline = (n - ii - 1)*vertical_offset
        ax.axhline(baseline, color = "lightgray", linewidth = 1.0, zorder = 0)

        scaled_intensity = spec.intensity*conc
        ax.plot(spec.ppm, scaled_intensity + baseline, label = f"{name} ({conc:.3f})",
                linewidth = linewidth, alpha = alpha)
        
    if show_sum and not mixture_on_top:
        ax.plot(mixture.ppm, mixture.intensity, label = "Mixture", color = "black",
                linewidth = 1.5)

    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")

    if invert_ppm:
        ax.invert_xaxis()

    ax.legend(loc = "upper right")  
    ax.set_title("Mixture Components")

    if show:
        plt.show()

    return ax


def plot_difference(reference: Spectrum, target: Spectrum, ax = None,
                    invert_ppm: bool = True, show: bool = True):
    """
    Plot target - reference.
    """

    if len(reference) != len(target):
        raise ValueError("Spectra must have same length.")
    
    if not np.allclose(reference.ppm, target.ppm):
        raise ValueError("ppm axes do not match.")
    
    diff = target.intensity - reference.intensity

    if ax is None:
        fig, ax = plt.subplots(figsize = (10, 4))

    ax.plot(reference.ppm, diff)
    ax.axhline(0, linestyle = "--", linewidth = 1.0)

    ax.set_xlabel("ppm")
    ax.set_ylabel("Difference")

    if invert_ppm:
        ax.invert_xaxis()

    ax.set_title("Difference Spectrum")

    if show:
        plt.show()

    return ax

def plot_stack(spectra: Sequence[Spectrum], offset: float = 1.0,
               invert_ppm: bool = True, show: bool = True):
    """
    Stacked spectrum plot.
    Useful for preprocessing comparasions.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    for ii, spec in enumerate(spectra):
        y = spec.intensity + ii*offset
        ax.plot(spec.ppm, y, label = spec.name)

    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity + offset")

    if invert_ppm:
        ax.invert_xaxis()

    ax.legend()

    if show:
        plt.show()

    return ax
