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
    intensity : np.ndarray, Real-value spectrum intesisities.
    name : str, Optiona spectrum name.
            default = None
    metadata : dict, Optional metada dictionary
    """
    ppm: np.ndarray 
    intensity: np.ndarray 
    name: str | None = None
    metadata: Dict[str, Any] = field(default_factory = dict)

    def __post_init__(self):
        self.ppm = np.asarray(self.ppm, dtype = float)
        self.intensity = np.asarray(self.intensity, dtype = np.float32)

        if len(self.ppm) != len(self.intensity):
            raise ValueError("ppm and intensity must have the same length.")

    @property
    def n_points(self) -> int:
        return len(self.ppm)
    
    @property
    def shape(self):
        return self.intensity.shape
    
    def copy(self) -> "Spectrum":
        return Spectrum(ppm = self.ppm.copy(),
                        intensity = self.intensity.copy(),
                        name = self.name,
                        metadata = self.metadata.copy())
    
    def max(self) -> float:
        return float(np.max(self.intensity))
    
    def min(self) -> float:
        return float(np.min(self.intensity))
    
    def area(self) -> float:
        return float(np.trapz(self.intensity, self.ppm))
    
    def normalize(self) -> "Spectrum":
        """
        Normalize by maximum absolute peak.
        """
        peak = np.max(np.abs(self.intensity))

        if peak == 0:
            return self.copy()
        
        return Spectrum(ppm = self.ppm.copy(), intensity = self.intensity/peak,
                        name = self.name, metadata = self.metadata.copy())
    
    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        (ppm, intensity)
        """
        return self.ppm, self.intensity
    
    def crop(self, ppm_min: float, ppm_max: float) -> "Spectrum":
        """
        Return cropped copy.
        """
        mask = (self.ppm >= ppm_min) & (self.ppm <= ppm_max)

        return Spectrum(ppm = self.ppm[mask], intensity = self.intensity[mask],
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

        ax.plot(self.ppm, self.intensity, **kwargs)

        if invert_ppm:
            ax.invert_xaxis()

        ax.set_xlabel("ppm")
        ax.set_ylabel("Intensity")

        if title is None:
            title = self.name
        if title is not None:
            ax.set_title(title)

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
    """
    composition: dict[str, float] = field(default_factory = dict)   

    @property
    def compounds(self) -> list[str]:
        return list(self.composition.keys())
    
    @property
    def n_compounds(self) -> list[str]:
        return list(self.composition)
    
    def copy(self) -> "MixtureSpectrum":
        return MixtureSpectrum(ppm = self.ppm.copy(), intensity = self.intensity.copy()
                               name = self.name, metadata = self.metadata.copy(),
                               composition = self.composition.copy())
    
    def summary(self):
        print(f"Mixture with {len(self.composition)} compounds.")
        for comp, conc in self.composition.items():
            print(f"{comp:<30} {conc:.4f}")

