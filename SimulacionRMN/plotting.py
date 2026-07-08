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
        if label is not None:
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
