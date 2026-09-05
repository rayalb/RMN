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

    @staticmethod
    def _check_ppm_compatibility(spectra: Sequence[Spectrum]) -> None:
        """
        Check that all spectra use the same ppm axis.
        """
        if len(spectra) == 0:
            raise ValueError("No spectra provided.")
        
        ref_ppm = spectra[0].ppm

        for spec in spectra[1:]:
            if len(spec.ppm) != len(ref_ppm):
                raise ValueError("Spectra have different lengths.")
            if not np.allclose(spec.ppm, ref_ppm, rtol = 0, atol = 1e-6):
                raise ValueError("Spectra are not aligned on the same ppm axis.")

    @staticmethod
    def _validate_concentrations(concentrations: Sequence[float]) -> np.ndarray:
        """
        Convert concentrations to a numpy array and validate them.
        """

        concentrations = np.asarray(concentrations, dtype = float)

        if np.any(~np.isfinite(concentrations)):
            raise ValueError(
                "Concetrations must be finite."
            )
        if np.any(concentrations < 0):
            raise ValueError("Concentrations cannot be negative.")
        if np.all(concentrations == 0):
            raise ValueError("At least one concentration must be positive.")
        return concentrations
        
    def simulate(self, compounds: Sequence[str], concentrations: Sequence[float],
                 normalize_concentrations: bool = False, store_components: bool = False) -> MixtureSpectrum:
        """
        Generates a mixture from specified compounds.

        Parameters
        ----------
            compounds : Names of compounds in the library.
            concentrations : Relative concentrations.
            normalize_concetrations : If True, concentrations are normalize to sum one.
            store_components: If True, store the individual spectra inside the 
                                resulting MixtureSpectrum.

        Returns
        -------
            MixtureSpectrum
        
        Example:
        --------
            mixture = simulator.simulate(compounds = ["Glucose", "Alanine],
                                        concentrations = [0.3, 0.7])
        """

        compounds = list(compounds)
        if len(compounds) == 0:
            raise ValueError("No compounds provided.")
        
        concentrations = self._validate_concentrations(concentrations)

        if len(compounds) != len(concentrations):
            raise ValueError("Compounds and concentrations must have the same length.")
        
        if normalize_concentrations:
            total = concentrations.sum()
            if total <= 0:
                raise ValueError("Concentration sum must be positive.")
            
            concentrations = concentrations/total

        spectra = [self.library.get_raw(comp) for comp in compounds]
        self._check_ppm_compatibility(spectra)

        has_imag = any(spec.imag is not None for spec in spectra)

        ppm = spectra[0].ppm.copy()
        mixture_real = np.zeros_like(spectra[0].real, dtype = float)
        mixture_imag = None
        if has_imag:
            mixture_imag = np.zeros_like(spectra[0].real, dtype = float)

        components = {}
        if store_components:
            components = {compound: spec.copy() for compound, spec in zip(compounds, spectra)}

       
        composition = {}
        for compound, concentration, spec in zip(compounds, concentrations, spectra):
            concentration = float(concentration)

            mixture_real += concentration*spec.real

            if has_imag:
                if spec.imag is not None:
                    mixture_imag += concentration*spec.imag

            composition[compound] = concentration 

        metadata = {"n_compounds": len(compounds), "has_imag": has_imag}
        simulator_metadata = {"normalize_concetrations": normalize_concentrations,
                              "store_components": store_components}
        
        return MixtureSpectrum(ppm = ppm, real = mixture_real.astype(np.float32),
                               imag = mixture_imag.astype(np.float32) if mixture_imag is not None else None,
                               name = "Mixture", 
                               metadata = metadata, composition = composition, components = components,
                               simulator_metadata = simulator_metadata)
    
    # Dictionary interface

    def simulate_from_dict(self, composition: dict[str, float],
                           normalize_concentrations: bool = False,
                           store_components: bool = False) -> MixtureSpectrum:
        """
        Generate mixture from a dictionary.
        Example
        -------
            mix = simulator.simulate_from_dict({
                    "Glucose": 0.4,
                    "Alanine": 0.6}, store_components = True
                )
        """
        return self.simulate(compounds = list(composition.keys()),
                             concentrations = list(composition.values()),
                             normalize_concentrations = normalize_concentrations,
                             store_components = store_components)
    
    
    def simulate_random(self, n_compounds: int | tuple[int, int] = (2, 8),
                        concentration_range: tuple[float, float] = (0.01, 1.0),
                        normalize_concentrations: bool = False, store_components: bool = False) -> MixtureSpectrum:
        """
        Generate a random mixture.
        
        Parameters
        ----------
            n_compounds : Number of compounds.
                            If tuple(min, max), a random number is sampled.
            concentration_range : Uniform sampling interval
        
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
                             normalize_concentrations = normalize_concentrations, store_components = store_components)
    
    def simulate_from_spectra(self, spectra: Sequence[Spectrum], concentrations: Sequence[float], 
                              names: Sequence[str] | None = None, normalize_concentrations: bool = False,
                              store_components: bool = True, metadata: dict | None = None) -> MixtureSpectrum:
        """
        Generate a mixture from Spectrum objects.
        Parameters
        ----------
            spectra: list[Spectrum], Spectra to mix.
            concentrations: list[float], relative concentrations.
            names: list[str], optional.
                    Names for the spectra. If None, spectrum.name is used.
            normalize_concentrations: bool, optional.
                                    Normalize concetrations so they sum to one.
            store_components: Store the component spectra.
            metadata: dict, optional.
                    Extra metadata for the mixture.
        """
        spectra = list(spectra)

        if len(spectra) == 0:
            raise ValueError("No spectra provided.")

        concentrations = self._validate_concentrations(concentrations)
        
        if len(spectra) != len(concentrations):
            raise ValueError("Number of spectra and concetrations must match.")
        
        if normalize_concentrations:
            total = concentrations.sum()
            if total <= 0:
                raise ValueError("Concentrations must sum to a positive value.")
            concentrations /= total

        if names is not None:
            names = list(names)
            if len(names) != len(spectra):
                raise ValueError("Number of names must match number of spectra.")

        self._check_ppm_compatibility(spectra)
        ppm = spectra[0].ppm.copy()

        has_imag = any(spec.imag is not None for spec in spectra)

        mixture_real = np.zeros_like(spectra[0].real, dtype = float)
        mixture_imag = None
        if has_imag:
            mixture_imag = np.zeros_like(spectra[0].real, dtype = float)

        composition = {}
        stored_components = {}

        for ii, (spec, concentration) in enumerate(zip(spectra, concentrations)):
            concentration = float(concentration)

            if names is None:
                name = spec.name or f"Component_{ii+1}"
            else:
                name = names[ii]

            mixture_real += concentration*spec.real
            if has_imag and spec.imag is not None:
                mixture_imag += concentration*spec.imag


            composition[name] = concentration
            if store_components:
                stored_components[name] = spec.copy()


        meta = dict(metadata or {})
        meta.update({"n_compounds": len(spectra),
                    "generated_from": "spectra"
        })
        simulator_metadata = {"normalized_concentrations": normalize_concentrations,
                              "store_components": store_components}
        
        return MixtureSpectrum(ppm = ppm, real = mixture_real.astype(np.float32),
                               imag = mixture_imag.astype(np.float32) if mixture_imag is not None else None,
                               name = "Mixture", composition = composition, components = stored_components,
                               metadata = meta, simulator_metadata = simulator_metadata)


    def generate_batch(self, n_mixtures: int, **kwargs) -> list[MixtureSpectrum]:
        """
        Generate multiple mixtures.
        """
        if n_mixtures < 1:
            raise ValueError("n_mixtures must be >=1.")

        return [self.simulate_random(**kwargs) for _ in range(n_mixtures)]
    
    def get_compounds(self, mixture: MixtureSpectrum) -> dict[str, Spectrum]:
        """
        Return the spectra corresponding to the compounds presetn in a
        mixture.
        Is the mixture already stores the spectra, they are returned directly.
        Otherwise, they are loaded from the library.
        """
        if mixture.components:
            return {name: spec.copy() for name, spec in mixture.components.items()}
        
        return {compound: self.library.get_raw(compound) for compound in mixture.composition}

    @staticmethod
    def summary(mixture: MixtureSpectrum):
        """
        Print composition.
        """
        print(f"Mixture with {len(mixture.composition)} compounds")

        for comp, conc in mixture.composition.items():
            print(f"{comp: < 25} {conc:.4f}")
        