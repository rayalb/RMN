"""
Simulacion RMN

Simulation and preprocessing tools for NMR spectra. 
This package provides a set of tools to simulate and preprocess NMR spectra, 
including functions for baseline correction, peak picking, and spectral alignment.
"""

from .spectrum import Spectrum, MixtureSpectrum
from .library import SpectrumLibrary
from .augmentation import SpectrumSimulator
from .mixture import MixtureSimulator

