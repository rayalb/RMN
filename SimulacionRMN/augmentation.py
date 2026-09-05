"""
augmentation.py
---------------
Spectrum augmentation utilities.
Applies realistic NMR distortions to Apectrum objects.

Implemented distortions
-------------------------
    - Chemical shift perturabtion.
    - Gaussian broadening.
    - Lorentizian broadiening.
    - Asymmetric line broadening.
    - Zero and first order phase distortion.
    - Constant baseline offset.
    - Additive Gaussian noise.
    - Global intensity scaling.

For Example:
    sim = SpectrumSimulator(random_state = 15)
    spec1 = sim.shift(spec)
    spec2 = sim.phase(spec)  
or
    spec_aug = sim(spec)
or
    spec_aug = sim.simulate(spec, apply_phase = False, apply_noise = False)
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
from scipy.signal import hilbert

from .spectrum import Spectrum


class SpectrumAugmentator:
    """
    NMR Spectrum simulator. Can simulate:
        - individual compounds.
        - synthetic mixtures.
        - experimental spectra.
    with shifts, broadening, asymmetry, phase distortion, baseline distortion, 
    additive noise.
    """
    def __init__(self, max_shift: int = 60,
                 gauss_sigma_range: Tuple[float, float] = (0.0, 2.0),
                 lorentz_gamma_range: Tuple[float, float] = (0.0, 2.0),
                 asym_prob: float = 0.2,
                 asym_decay_pts: float = 30.0,
                 asym_amp_range: Tuple[float, float] = (0.0, 0.10),
                 phi0_range: Tuple[float, float] = (-15.0, 15.0),
                 phi1_range: Tuple[float, float] = (-3.0, 3.0),
                 baseline_range: Tuple[float, float] = (-0.02, 0.02),
                 noise_fraction: float = 0.02,
                 intensity_scale_range: Tuple[float, float] = (0.95, 1.05), 
                 random_state: Optional[int] = None):
        self.max_shift = max_shift
        self.gauss_sigma_range = gauss_sigma_range
        self.lorentz_gamma_range = lorentz_gamma_range
        self.asym_prob = asym_prob
        self.asym_decay_pts = asym_decay_pts
        self.asym_amp_range = asym_amp_range
        self.phi0_range = phi0_range
        self.phi1_range = phi1_range
        self.baseline_range = baseline_range
        self.intensity_scale_range = intensity_scale_range
        self.noise_fraction = noise_fraction

        self.rng = np.random.default_rng(random_state)

    def _shift_array(self, arr: np.ndarray, shift_pts: int) -> np.ndarray:
        """
        Shift an array by a number of points, filling the empty space with zeros.
        """
        if shift_pts == 0:
            return arr.copy()
        shifted = np.roll(arr, shift_pts)
        if shift_pts > 0:
            shifted[:shift_pts] = 0.0
        elif shift_pts < 0:
            shifted[shift_pts:] = 0.0
        return shifted

    def _convolve_array(self, arr: np.ndarray, kernels: List[np.ndarray]) -> np.ndarray:
        """
        Convolve an array with a kernel, using 'same' mode.
        """
        out= arr.copy()
        for kk in kernels:
            out = np.convolve(out, kk, mode='same')
        return out  
    
    def shift(self, spectrum: Spectrum) -> Spectrum:
        
        shift_pts = int(self.rng.normal(loc = 0, scale = self.max_shift/3))
        shift_pts = int(np.clip(shift_pts, -self.max_shift, self.max_shift))

        # shifted = ng.process.porc_base.cs(y, shift_pts)
        # TODO: CHEQUEAR si hace lo mismo que el nmrglue.
        real = self._shift_array(spectrum.real, shift_pts)
        imag = None
        if spectrum.imag is not None:
            imag = self._shift_array(spectrum.imag, shift_pts)  
        metadata = dict(spectrum.metadata)    
        metadata["shift_pts"] = shift_pts

        return Spectrum(ppm = spectrum.ppm.copy(), real = real.astype(np.float32),
                        imag = imag.astype(np.float32) if imag is not None else None,
                        name = spectrum.name, metadata = metadata)
    
    
    def broaden(self, spectrum: Spectrum) -> Spectrum:

        metadata = dict(spectrum.metadata)
        kernels = []

        # Gaussian broadening
        sigma = self.rng.uniform(*self.gauss_sigma_range)
        metadata["gauss_sigma"] = float(sigma)
        if sigma > 0:
            L = max(3, int(6*sigma) + 1)
            x = np.arange(-(L//2), L//2 + 1)
            g = np.exp(-0.5*(x/sigma)**2)
            kernels.append(g/g.sum())

        # Lorentzian broadening
        gamma = self.rng.uniform(*self.lorentz_gamma_range)
        metadata["lorentz_gamma"] = float(gamma)
        if gamma > 0:
            L = max(3, int(10*gamma) + 1)
            x = np.arange(-(L//2), L//2 + 1)
            l = 1.0/(1.0 + (x/gamma)**2)
            kernels.append(l/l.sum())

        # Asymmetric tail
        metadata["asym_applied"] = False
        if self.rng.random() < self.asym_prob and self.asym_decay_pts > 0:
            amp = self.rng.uniform(*self.asym_amp_range)
            direction = int(self.rng.choice([-1, 1]))

            metadata["asym_applied"] = True
            metadata["asym_amp"] = float(amp)
            metadata["asym_dir"] = direction

            L = max(5, int(5*self.asym_decay_pts))
            decay = np.exp(-np.arange(L)/self.asym_decay_pts)
            decay /= decay.sum()

            asym_kernel = np.zeros(2*L+1)
            asym_kernel[L] = 1.0 - amp
            if direction > 0:
                asym_kernel[L+1:L + 1 + L] = amp*decay    # tail to the right
            else:
                asym_kernel[L - L: L] = (amp*decay)[::-1] # tail to the left
            kernels.append(asym_kernel)

        real = self._convolve_array(spectrum.real, kernels)
        imag = None
        if spectrum.imag is not None:
            imag = self._convolve_array(spectrum.imag, kernels)
        metadata["broadened"] = True

        return Spectrum(ppm = spectrum.ppm.copy(), real = real.astype(np.float32),
                        imag = imag.astype(np.float32) if imag is not None else None,
                        name = spectrum.name, metadata = metadata)
    
    def phase(self, spectrum: Spectrum) -> Spectrum:
        phi0 = self.rng.uniform(*self.phi0_range)
        phi1 = self.rng.uniform(*self.phi1_range)

        ref_ppm = (spectrum.ppm.max() + spectrum.ppm.min())/2
        phase_rad = np.deg2rad(phi0 + phi1*(spectrum.ppm - ref_ppm))
        phase_factor = np.exp(1j*phase_rad)

        complex_spectrum = spectrum.real + 1j*(spectrum.imag if spectrum.imag is not None 
                                               else hilbert(spectrum.real).imag)
        phased = complex_spectrum*phase_factor

        metadata = dict(spectrum.metadata)
        metadata["phi0"] = float(phi0)
        metadata["phi1"] = float(phi1)

        return Spectrum(ppm = spectrum.ppm.copy(),
                        real = phased.real.astype(np.float32),
                        imag = phased.imag.astype(np.float32) if phased.imag is not None else None,
                        name = spectrum.name, metadata = metadata)
    
    def baseline(self, spectrum: Spectrum) -> Spectrum:
        offset = self.rng.uniform(*self.baseline_range)
        metadata = dict(spectrum.metadata)
        metadata["baseline_offset"] = float(offset)

        return Spectrum(ppm = spectrum.ppm.copy(), real = (spectrum.real.copy() + offset).astype(np.float32),
                        imag = (spectrum.imag.copy() + offset).astype(np.float32) if spectrum.imag is not None else None,
                        name = spectrum.name, metadata = metadata)
    
    def noise(self, spectrum: Spectrum) -> Spectrum:
        metadata = dict(spectrum.metadata)

        if spectrum.imag is not None:
            rms = np.sqrt(np.mean(spectrum.real**2 + spectrum.imag**2))
            sigma = self.noise_fraction*rms
            channel_sigma = sigma/np.sqrt(2)
            real_noise = self.rng.normal(loc = 0.0, scale = channel_sigma, size = len(spectrum.real))
            imag_noise = self.rng.normal(loc = 0.0, scale = channel_sigma, size = len(spectrum.imag))
            real = spectrum.real + real_noise
            imag = spectrum.imag + imag_noise
        else:
            # Real part only
            rms = np.sqrt(np.mean(spectrum.real**2))
            sigma = self.noise_fraction*rms
            real = spectrum.real + self.rng.normal(loc = 0.0, scale = sigma, size = len(spectrum.real))
            imag = None

        metadata["noise_sigma"] = float(sigma)
        return Spectrum(ppm = spectrum.ppm.copy(), real = real.astype(np.float32),
                        imag = imag.astype(np.float32) if imag is not None else None,
                        name = spectrum.name, metadata = metadata)

    def scale(self, spectrum: Spectrum) -> Spectrum:
        factor = self.rng.uniform(*self.intensity_scale_range)

        real = factor * spectrum.real
        imag = None
        if spectrum.imag is not None:
            imag = factor * spectrum.imag

        metadata = dict(spectrum.metadata)
        metadata["scale"] = float(factor)
        return Spectrum(ppm = spectrum.ppm.copy(), 
                        real = real.astype(np.float32), 
                        imag = imag.astype(np.float32) if imag is not None else None,
                        name = spectrum.name, metadata = metadata)
    
    def simulate(self, spectrum: Spectrum, **kwargs) -> Spectrum:
        """
        Simulate distortion on a spectrum.
        Returns
        -------
        Spectrum: Augmented spectrum.
        """
        spec = spectrum.copy()
        methods = [("apply_shift", self.shift), ("apply_broadening", self.broaden),
                   ("apply_phase", self.phase), ("apply_baseline", self.baseline),
                   ("apply_noise", self.noise), ("apply_scaling", self.scale)]
        for flag, method in methods:
            if kwargs.get(flag, True):
                spec = method(spec)
        return spec 
    
    def __call__(self, spectrum: Spectrum, **kwargs):
        return self.simulate(spectrum, **kwargs)
    

   
    
    
    
    
    

