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

from typing import Optional, Tuple

import numpy as np
from scipy.signal import hilbert

from .spectrum import Spectrum


class SpectrumSimulator:
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


    def shift(self, spectrum: Spectrum) -> Spectrum:
        y = spectrum.intensity.copy()
        metadata = dict(spectrum.metadata)

        shift_pts = int(self.rng.normal(loc = 0, scale = self.max_shift/3))
        shift_pts = int(np.clip(shift_pts, -self.max_shift, self.max_shift))

        # shifted = ng.process.porc_base.cs(y, shift_pts)
        # TODO: CHEQUEAR si hace lo mismo que el nmrglue.
        shifted = np.roll(y, shift_pts)
        if shift_pts > 0:
            y[:shift_pts] = 0.0
        elif shift_pts < 0:
            y[shift_pts:] = 0.0
        
        metadata["shift_pts"] = shift_pts

        return Spectrum(ppm = spectrum.ppm.copy(), intensity = y.astype(np.float32),
                        name = spectrum.name, metadata = metadata)
    
    
    def broaden(self, spectrum: Spectrum) -> Spectrum:
        y = spectrum.intensity.copy()
        metadata = dict(spectrum.metadata)

         # Gaussian broadening
        sigma = self.rng.uniform(*self.gauss_sigma_range)
        metadata["gauss_sigma"] = float(sigma)
        if sigma > 0:
            L = max(3, int(6*sigma) + 1)
            x = np.arange(-(L//2), L//2 + 1)
            g = np.exp(-0.5*(x/sigma)**2)
            g /= g.sum()
            y = np.convolve(y, g, mode = 'same')

        # Lorentzian broadening
        gamma = self.rng.uniform(*self.lorentz_gamma_range)
        metadata["lorentz_gamma"] = float(gamma)
        if gamma > 0:
            L = max(3, int(10*gamma) + 1)
            x = np.arange(-(L//2), L//2 + 1)
            l = 1.0/(1.0 + (x/gamma)**2)
            l /= l.sum()
            y = np.convolve(y, l, mode = 'same')
        
         # Asymmetric tail
        metadata["asym_applied"] = False

        if self.rng.random() < self.asym_prob and self.asym_decay_pts > 0:
            amp = self.rng.uniform(*self.asym_amp_range)
            direction = int(self.rng.choice([-1, 1]))

            metadata["asym_applied"] = True
            metadata["asym_amp"] = float(amp)
            metadata["asym_dir"] = direction

            L = max(5, int(5*self.asym_decay_pts))
            x = np.arange(L)
            decay = np.exp(-x/self.asym_decay_pts)
            decay /= decay.sum()

            kernel = np.zeros(2*L+1)
            kernel[L] = 1.0 - amp
            if direction > 0:
                kernel[L+1:L + 1 + L] = amp*decay    # tail to the right
            else:
                kernel[L - L: L] = (amp*decay)[::-1] # tail to the left

            y = np.convolve(y, kernel, mode = "same")

        return Spectrum(ppm = spectrum.ppm.copy(), intensity = y.astype(np.float32),
                        name = spectrum.name, metadata = metadata)
    
    def phase(self, spectrum: Spectrum) -> Spectrum:
        phi0 = self.rng.uniform(*self.phi0_range)
        phi1 = self.rng.uniform(*self.phi1_range)

        ref_ppm = (spectrum.ppm.max() + spectrum.ppm.min())/2
        phase = np.deg2rad(phi0 + phi1*(spectrum.ppm - ref_ppm))
        analytic = hilbert(spectrum.intensity)
        analytic *= np.exp(1j*phase)
        metadata = dict(spectrum.metadata)

        metadata["phi0"] = float(phi0)
        metadata["phi1"] = float(phi1)

        return Spectrum(ppm = spectrum.ppm.copy(),
                        intensity = analytic.real.astype(np.float32),
                        name = spectrum.name, metadata = metadata)
    
    def baseline(self, spectrum: Spectrum) -> Spectrum:
        offset = self.rng.uniform(*self.baseline_range)
        y = spectrum.intensity + offset
        metadata = dict(spectrum.metadata)
        metadata["baseline_offset"] = float(offset)

        return Spectrum(ppm = spectrum.ppm.copy(), intensity = y.astype(np.float32),
                        name = spectrum.name, metadata = metadata)
    
    def noise(self, spectrum: Spectrum) -> Spectrum:
        rms = np.sqrt(np.mean(spectrum.intensity**2))
        sigma = self.noise_fraction*rms
        noise = self.rng.normal(loc = 0.0, scale = sigma, size = len(spectrum))
        y = spectrum.intensity + noise 
        metadata = dict(spectrum.metadata)
        metadata["noise_sigma"] = float(sigma)

        return Spectrum(ppm = spectrum.ppm.copy(), intensity = y.astype(np.float32),
                        name = spectrum.name, metadata = metadata)
    
    def scale(self, spectrum: Spectrum) -> Spectrum:
        factor = self.rng.uniform(*self.intensity_scale_range)

        metadata = dict(spectrum.metadata)
        metadata["scale"] = float(factor)
        return Spectrum(ppm = spectrum.ppm.copy(), 
                        intensity = (factor * spectrum.intensity).astype(np.float32),
                        name = spectrum.name, metadata = metadata)
    
    def simulate(self, spectrum: Spectrum, apply_shift: bool = True,
                 apply_broadening: bool = True,
                 apply_phase: bool = True,
                 apply_baseline: bool = True,
                 apply_noise: bool = True,
                 apply_scaling: bool = True) -> Spectrum:
        """
        Simulate distortion on a spectrum.
        Returns
        -------
        Spectrum: Augmented spectrum.
        """
        spec = spectrum.copy()

        # Shift
        if apply_shift:
            spec = self.shift(spec)
        # Broadening + assymmetry
        if apply_broadening:
            spec = self.broaden(spec)
        # Phase
        if apply_phase:
            spec = self.phase(spec)
        # Baseline
        if apply_baseline:
            spec = self.baseline(spec)
        # Noise
        if apply_noise:
            spec = self.noise(spec)
        # Scaling
        if apply_scaling:
            spec = self.scale(spec)

        return spec 
    
    def __call__(self, spectrum: Spectrum, **kwargs):
        return self.simulate(spectrum, **kwargs)
    

   
    
    
    
    
    

