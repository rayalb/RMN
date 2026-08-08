"""
plotting.py
-----------

Visualization utilities for NMR spectra.

Supports:
* Single spectrum (real part, imaginary part, or magnitude, phase)
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


def _get_plot_data(spectrum: Spectrum, mode: str) -> np.ndarray:
    """
    Get the data to plot based on the selected mode.
    """
    mode = mode.lower()
    if mode == "real":
        return spectrum.real, "Real Part"
    elif mode == "imag":
        if spectrum.imag is None:
            raise ValueError("Spectrum does not have an imaginary part.")
        return spectrum.imag, "Imaginary Part"
    elif mode == "magnitude":
        return spectrum.magnitude, "Magnitude"
    elif mode == "phase":
        return spectrum.phase, "Phase"
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose from 'real', 'imag', 'magnitude', or 'phase'.")



def plot_spectrum(spectrum: Spectrum, mode = "real", ax = None, title: str | None = None,
                  invert_ppm: bool = True, linewidth: float = 1.0,
                  show: bool = True):
    """
    Plot a single spectrum.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize = (10, 4))

    if mode == "complex":
        ax.plot(spectrum.ppm, spectrum.real, linewidth = linewidth,
                label = "Real Part")
        if spectrum.imag is not None:
            ax.plot(spectrum.ppm, spectrum.imag, linewidth = linewidth,
                    label = "Imaginary Part")
        ylabel = "Spectrum intensity"
    else:
        data, ylabel = _get_plot_data(spectrum, mode)
        ax.plot(spectrum.ppm, data, linewidth = linewidth,
                label = spectrum.name)
    ax.set_xlabel("ppm")
    ax.set_ylabel(ylabel)

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

def plot_spectra(spectra: Sequence[Spectrum], mode: str = "real", labels: Sequence[str] | None = None, 
                 ax = None, title: str | None = None,
                 invert_ppm: bool = True, linewidth: float = 1.0,
                 alpha: float = 0.9, show: bool = True):
    """
    Overlay multiple spectra.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize = (10, 4))

    ylabel = None

    for ii, spec in enumerate(spectra):
        label = labels[ii] if labels is not None else spec.name
        data, ylabel = _get_plot_data(spec, mode)

        ax.plot(spec.ppm, data, linewidth = linewidth, alpha = alpha,
                label = label)

    ax.set_xlabel("ppm")
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if invert_ppm:
        ax.invert_xaxis()

    ax.legend()

    if show:
        plt.show()

    return ax

def plot_mixture(mixture: MixtureSpectrum, mode = "real", show_composition: bool = True,
                   ax = None, show: bool = True):
    """
    Plot mixture spectrum.
    """

    title = "Synthetic Mixture"

    if show_composition:
        comps = ", ".join(mixture.composition.keys())
        title += f"\n{comps}"

    return plot_spectrum(mixture, mode = mode, ax = ax, title = title, show = show)

def plot_mixture_components(mixture: MixtureSpectrum, library: SpectrumLibrary | None = None, 
                            mode: str = "real", show_sum: bool = True, mixture_on_top: bool = True, 
                            vertical_offset: float | None = None,
                            invert_ppm: bool = True, linewidth: float = 1.2,
                            alpha: float = 0.7, show: bool = True):
    """
    plot every component of a mixture, optionally with the sum spectrum.

    Parameters
    ----------
        mixture: MixtureSpectrum, final simalted mixture.
        library: SpectrumLibrary, optional, library containing the component spectra.
        mode: str, optional, the mode of the spectrum to plot.
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

    max_height = 0
    for name, spec in components.items():
        data, _ = _get_plot_data(spec, mode)
        max_height = max(max_height, np.max(np.abs(data*mixture.composition[name])))

    if vertical_offset is None:
        vertical_offset = 1.3*max_height

    n = len(components)
    if show_sum and mixture_on_top:
        data_mix, _ = _get_plot_data(mixture, mode)
        baseline = n*vertical_offset
        ax.plot(mixture.ppm, data_mix + baseline, label = "Mixture",
                linewidth = 1.5, color = "black")

    for ii, (name, spec) in enumerate(components.items()):
        data, _ = _get_plot_data(spec, mode)
        baseline = (n - ii - 1)*vertical_offset
        ax.axhline(baseline, color = "lightgray", linewidth = 1.0)
        ax.plot(spec.ppm, data*mixture.composition[name] + baseline, linewidth = linewidth, alpha = alpha,
                label = f"{name} ({mixture.composition[name]:.3f})")

    if show_sum and not mixture_on_top:
        data_mix, _ = _get_plot_data(mixture, mode)
        ax.plot(mixture.ppm, data_mix, label = "Mixture", color = "black",
                linewidth = 1.5)

    ax.set_xlabel("ppm")
    ax.set_ylabel(mode)
    
    if invert_ppm:
        ax.invert_xaxis()

    ax.legend(loc = "upper right")  
    ax.set_title("Mixture Components")

    if show:
        plt.show()

    return ax


def plot_difference(reference: Spectrum, target: Spectrum, mode = "real", ax = None,
                    invert_ppm: bool = True, show: bool = True):
    """
    Plot target - reference.
    """

    if len(reference) != len(target):
        raise ValueError("Spectra must have same length.")
    
    if not np.allclose(reference.ppm, target.ppm):
        raise ValueError("ppm axes do not match.")

    y_ref, ylabel = _get_plot_data(reference, mode)
    y_target, _ = _get_plot_data(target, mode)

    if ax is None:
        fig, ax = plt.subplots(figsize = (10, 4))

    ax.plot(reference.ppm, y_target - y_ref,linewidth = 1.5)
    ax.axhline(0, linestyle = "--", linewidth = 1.0)

    ax.set_xlabel("ppm")
    ax.set_ylabel(f"$\Delta${ylabel}$")

    if invert_ppm:
        ax.invert_xaxis()

    ax.set_title("Difference Spectrum")

    if show:
        plt.show()

    return ax

def plot_stack(spectra: Sequence[Spectrum], mode = "real", offset: float = 1.0,
               invert_ppm: bool = True, show: bool = True):
    """
    Stacked spectrum plot.
    Useful for preprocessing comparasions.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    for ii, spec in enumerate(spectra):
        data, ylabel = _get_plot_data(spec, mode)
        y = data + ii*offset
        ax.plot(spec.ppm, y, label = spec.name)

    ax.set_xlabel("ppm")
    ax.set_ylabel(f"{ylabel} + offset")

    if invert_ppm:
        ax.invert_xaxis()

    ax.legend()

    if show:
        plt.show()

    return ax

def plot_complex(spectra: Spectrum, invert_ppm: bool = True, linewidth: float = 1.2, show: bool = True):
    """
    Plot real and imaginary, magnitude and phase in four aligned panels.
    """

    if spectra.imag is None:
        raise ValueError("Spectrum has no imaginary component")
    
    fig, axes = plt.subplots(2, 2, figsize = (12, 8), sharex = True)
    plots = [
        ("Real", spectra.real),
        ("IMaginary", spectra.imag),
        ("Magnitude", spectra.magnitude),
        ("Phase", spectra.phase)
    ]

    for ax, (title, data) in zip(axes.ravel(), plots):
        ax.plot(spectra.ppm, data, linewidth = linewidth)
        ax.set_title(title)
        ax.set_xlabel("ppm")
        ax.set_ylabel(title)
        
        if invert_ppm:
            ax.invert_xaxis()
        ax.grid(True, alpha = 0.3)

    plt.tight_layout()
    if show:
        plt.show()

    return axes     