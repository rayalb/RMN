"""
spectrum.py
-----------

Core spectrum containers.
Defines:
* Spectrum
* MixtureSpectrum

Used throughout the package.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Spectrum:
    """
    Spectrum Container.
    Parameters
    ----------
    ppm: np.ndarray, Chemical shift axis.
    real: np.ndarray, real-part  of the spectrum.
    img: np.ndarray, imaginary-part of the spectrum.
            If None, the spectrum is considered real.    
    name : str, Optiona spectrum name.
            default = None
    metadata : dict, Optional metada dictionary
    """
    ppm: np.ndarray 
    real: np.ndarray
    imag: np.ndarray | None = None 
    name: str | None = None
    metadata: Dict[str, Any] = field(default_factory = dict)

    def __post_init__(self):
        self.ppm = np.asarray(self.ppm, dtype = float)
        self.real = np.asarray(self.real, dtype = np.float32)

        if self.imag is not None:
            self.imag = np.asarray(self.imag, dtype = np.float32)
            if len(self.ppm) != len(self.imag):
                raise ValueError("ppm and imag must have the same length.")
        
        if len(self.ppm) != len(self.real):
            raise ValueError("ppm and real must have the same length.")
        
    @property
    def intensity(self) -> np.ndarray:
        return self.real
    
    @property
    def complex(self) -> np.ndarray:
        if self.imag is None:
            return self.real.astype(np.complex64)
        else:
            return self.real.astype(np.complex64) + 1j*self.imag.astype(np.complex64)

    @property
    def magnitude(self) -> np.ndarray:
        if self.imag is None:
            return np.abs(self.real) 
        return np.abs(self.complex)

    @property
    def phase(self) -> np.ndarray:
        if self.imag is None:
            return np.zeros_like(self.real)
        return np.angle(self.complex)
    
    @property
    def n_points(self) -> int:
        return len(self.ppm)
    
    @property
    def shape(self):
        return self.real.shape
    
    def copy(self) -> "Spectrum":
        return Spectrum(ppm = self.ppm.copy(),
                        real = self.real.copy(),
                        imag = None if self.imag is None else self.imag.copy(),
                        name = self.name,
                        metadata = self.metadata.copy())
    
    def max(self) -> float:
        return float(np.max(self.real))
    
    def min(self) -> float:
        return float(np.min(self.real))
    
    def area(self) -> float:
        return float(np.trapzoid(self.real, self.ppm))
    
    def normalize(self) -> "Spectrum":
        """
        Normalize by maximum absolute peak.
        """
        peak = np.max(np.abs(self.complex))

        if peak == 0:
            return self.copy()
        
        return Spectrum(ppm = self.ppm.copy(), real = self.real/peak,
                        imag = None if self.imag is None else self.imag/peak,
                        name = self.name, metadata = self.metadata.copy())
    
    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        (ppm, intensity)
        """
        return self.ppm, self.real, self.imag
    
    def crop(self, ppm_min: float, ppm_max: float) -> "Spectrum":
        """
        Return cropped copy.
        """
        mask = (self.ppm >= ppm_min) & (self.ppm <= ppm_max)

        return Spectrum(ppm = self.ppm[mask], real = self.real[mask],
                        imag = None if self.imag is None else self.imag[mask],
                        name = self.name, metadata = self.metadata.copy())

    def plot(self, ax = None, figsize = (10, 4), invert_ppm: bool = True,
             title: Optional[str] = None, **kwargs):
        """
        Plot spectrum.
        Parameters
        ----------
            invert_ppm : bool. NMR convention uses decreasing ppm.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)

        ax.plot(self.ppm, self.real, **kwargs)

        if invert_ppm:
            ax.invert_xaxis()

        ax.set_xlabel("ppm")
        ax.set_ylabel("Intensity")

        if title is None:
            title = self.name
            ax.set_title(title)
        if title is not None:
            ax.set_title(title)

        plt.show()
        return ax


    def __len__(self):
        return len(self.ppm)
    
    def __repr__(self):
        name = self.name if self.name else "Unnamed"
        return(f"Spectrum("
               f"name = '{name}', "
               f"points = {len(self)}, "
               f"range = [{self.ppm.min():.3f}, "
               f"{self.ppm.max():.3f}] ppm)")

@dataclass
class MixtureSpectrum(Spectrum):
    """
    Spectrum generated from multiple compounds.

    Parameters
    ----------
    composition : dict[str, float], Dictionary with compound names and their concentrations.
    components : dict[str, Spectrum], Dictionary with compound names and their corresponding spectra.
    simulator_metadata : dict[str, Any], Optional metadata from the simulator.
    """
    composition: dict[str, float] = field(default_factory = dict)   
    components: dict[str, Spectrum] = field(default_factory = dict)
    simulator_metadata: dict[str, Any] = field(default_factory = dict)
    
    @property
    def compounds(self) -> list[str]:
        return list(self.composition.keys())
    
    @property
    def n_compounds(self) -> list[str]:
        return list(self.composition)
    
    def copy(self) -> "MixtureSpectrum":
        return MixtureSpectrum(ppm = self.ppm.copy(), real = self.real.copy(),
                               imag = None if self.imag is None else self.imag.copy(),
                               name = self.name, metadata = self.metadata.copy(),
                               composition = self.composition.copy(),
                               components = {k: v.copy() for k, v in self.components.items()},
                               simulator_metadata = self.simulator_metadata.copy())
    
    def summary(self):
        print(f"Mixture with {len(self.composition)} compounds.")
        for comp, conc in self.composition.items():
            print(f"{comp:<30} {conc:.4f}")

