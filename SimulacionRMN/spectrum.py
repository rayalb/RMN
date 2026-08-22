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

        self.metadata.setdefault("has_imag", self.imag is not None)
        self.metadata.setdefault("n_points", len(self.ppm))

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
    
    def max(self, component: str = "real") -> float:
        """Maximum value of a selected spectra representation."""
        values = self._get_component(component)
        return float(np.max(values))
    
    def min(self, component: str = "real") -> float:
        """Minimum value of a selected spectra representation."""
        values = self._get_component(component)
        return float(np.min(values))
    
    def area(self, component: str = "real") -> float:
        """Numerical integral of a selected  spectra representation."""
        values = self._get_component(component)
        return float(np.trapzoid(values, self.ppm))

    def _get_component(self, component: str) -> np.ndarray:

        if component == "real":
            return self.real

        if component == "imag":
            if self.imag is None:
                return np.zeros_like(self.real)
            return self.imag

        if component in ("absolute", "magnitude", "abs"):
            return self.absolute

        if component == "phase":
            return self.phase

        raise ValueError(f"Unknown component '{component}'."
                         "Use 'real', 'imag', 'abs', or 'phase'.")
    
    def normalize(self, component: str = "abs") -> "Spectrum":
        """
        Normalize the spectrum by its maximum absolute value.
        component: str. Representation used to determine the normalization
                        factor. "real", "imag", "abs". Default: "abs"
        """
        values = self._get_component(component)

        peak = np.max(np.abs(values))

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

    def plot(self, ax = None, figsize = (10, 4), component: str = "real",
            invert_ppm: bool = True, title: Optional[str] = None, **kwargs):
        """
        Plot spectrum.
        Parameters
        ----------
            component : str. Representation of the spectrum, "real", "imag",
                                "abs", "phase". Default, "real".
            invert_ppm : bool. NMR convention uses decreasing ppm.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)

        values = self._get_component(component)
        ax.plot(self.ppm, values, **kwargs)

        if invert_ppm:
            ax.invert_xaxis()

        ax.set_xlabel("ppm")
        ylabel = {"real": "Real", "imag": "Imaginary", "absolute": "Absolute",
                  "magnitud": "Absolute", "abs": "Absolute", "phase": "Phase (rad)"
                  }.get(component, component)
        
        ax.set_ylabel(ylabel)

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
    Spectrum generated from multiple compounds. The mixture itself is represented
    using real and optional imaginary components.

    Parameters
    ----------
    composition : dict[str, float], Dictionary with compound names and their concentrations.
    components : dict[str, Spectrum], Dictionary with compound names and their corresponding spectra.
    simulator_metadata : dict[str, Any], Optional metadata from the simulator.
    internal_standard : Spectrum used as the internal standard, if presents.
    imternal_standard_scale : Scaling factor applied to the internal standard.
    """
    composition: dict[str, float] = field(default_factory = dict)   
    components: dict[str, Spectrum] = field(default_factory = dict)
    simulator_metadata: dict[str, Any] = field(default_factory = dict)
    internal_standard: Spectrum | None = None
    internal_standard_scale: float = 1.0

    @property
    def compounds(self) -> list[str]:
        return list(self.composition.keys())
    
    @property
    def n_compounds(self) -> list[str]:
        return list(self.composition)

    @property
    def has_internal_standard(self) -> bool:
        return self.internal_standard is not None
    
    def copy(self) -> "MixtureSpectrum":
        return MixtureSpectrum(ppm = self.ppm.copy(), real = self.real.copy(),
                               imag = None if self.imag is None else self.imag.copy(),
                               name = self.name, metadata = self.metadata.copy(),
                               composition = self.composition.copy(),
                               components = {name: spec.copy() for name, spec in self.components.items()},
                               simulator_metadata = self.simulator_metadata.copy(),
                               internal_standard = None if self.internal_standard is None else self.internal_standard.copy(),
                               internal_standard_scale = self.internal_standard_scale)
    
    def summary(self):
        print(f"Mixture with {len(self.composition)} compounds.")
        for comp, conc in self.composition.items():
            print(f"{comp:<30} {conc:.4f}")

        if self.internal_standard is not None:
            print(f"{'Internal standard':<30}"
                  f"{self.internal_standard.name}"
                  f"(scale = {self.internal_standard_scale:.4f})")
