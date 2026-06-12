"""
mixture.py
==========

Mixture simulation utilities.

Builds synthetic mixtures from spectra stored inside
a SpectrumLibrary

Returns MixtureSpectrum objects
"""

from __future__ import annotations
from typing import Sequence

import numpy as np

from .spectrum import Spectrum, MixtureSpectrum
from .library import SpectrumLibrary



class MixtureSimulator:
    """
    Synthetic NMR mixture generator.
    
    Parameters
    ----------
    library : SpectrumLibrary
        Library containing individual compound spectra.
    random_state: int | None
        Seed for reproducibility.
    """
    def __init__(self, library: SpectrumLibrary, random_state: int | None = None):
        self.library = library
        self.rng = np.random.default_rng(random_state)

    def _check_ppm_compatibility(self, spectra: list[Spectrum]):
        ref_ppm = spectra[0].ppm

        for spec in spectra[1:]:
            if len(spec.ppm) != len(ref_ppm):
                raise ValueError("Spectra have different lengths.")
            if not np.allclose(spec.ppm, ref_ppm):
                raise ValueError("Spectra are not aligned on the same ppm axis.")
            
    def simulate(self, compounds: Sequence[str], concentrations: Sequence[float],
                 normalize_concentrations: bool = False) -> MixtureSpectrum:
        """
        Generates a mixture from specified compounds.
        
        Example:
        --------
            mixture = simulator.simulate(compounds = ["Glucose", "Alanine],
                                        concentrations = [0.3, 0.7])
        """

        compounds = list(compounds)
        concentrations = np.asarray(concentrations, dtype = float)

        if len(compounds) != len(concentrations):
            raise ValueError("Compounds and concentrations must have the same length.")
        
        if normalize_concentrations:
            total = concentrations.sum()
            if total <= 0:
                raise ValueError("Concentration sum must be positive.")
            
            concentrations = concentrations/total

        spectra = [self.library.get_raw(comp) for comp in compounds]

        self._check_ppm_compatibility(spectra)

        ppm = spectra[0].ppm.copy()

        mixture = np.zeros_like(spectra[0].intensity, dtype = float)

        composition = {}
        for comp, conc, spec in zip(compounds, concentrations, spectra):
            mixture += conc*spec.intensity

            composition[comp] = float(conc)

        return MixtureSpectrum(ppm = ppm, intensity = mixture.astype(np.float32),
                               metadata = {"n_compounds": len(compounds)},
                               composition = composition)
    
    def simulate_random(self, n_compounds: int | tuple[int, int] = (2, 8),
                        concentration_range: tuple[float, float] = (0.01, 1.0),
                        normalize_concentrations: bool = False) -> MixtureSpectrum:
        """
        Generate a random mixture.
        
        Example:
        --------
        mix = simulator.simulate_random()
        """

        available = self.library.compounds()
        if len(available) == 0:
            raise RuntimeError("Spectrum library is empty.")
        
        if isinstance(n_compounds, tuple):
            n = self.rng.integers(n_compounds[0], n_compounds[1] + 1)
        else:
            n = int(n_compounds)

        n = min(n, len(available))

        compounds = self.rng.choice(available, size = n, replace = False)

        concentrations = self.rng.uniform(concentration_range[0], 
                                         concentration_range[1], size = n)
        return self.simulate(compounds = compounds, concentrations = concentrations,
                             normalize_concentrations = normalize_concentrations)
    
    def generate_batch(self, n_mixtures: int, **kwargs) -> list[MixtureSpectrum]:
        """
        Generate multiple mixtures.
        """
        mixtures = []
        for _ in range(n_mixtures):
            mixtures.append(self.simulate_random(**kwargs))

        return mixtures
    
    def summary(self, mixture: MixtureSpectrum):
        """
        Print composition.
        """
        print(f"Mixture with {len(mixture.composition)} compounds")

        for comp, conc in mixture.composition.item():
            print(f"{comp: < 25} {conc:.4f}")
        