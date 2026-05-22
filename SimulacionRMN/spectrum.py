from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any


import numpy as np
from scipy.signal import hilbert

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
    name = Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory = dict)

    def copy(self):
        return Spectrum(ppm = self.ppm.copy(),
                        intensity = self.intensity.copy(),
                        name = self.name,
                        metadata = self.metadata.copy())
    

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
                 asym_decay_pts: float = 30,
                 asym_amp_range: Tuple[float, float] = (0.0, 0.10),
                 phi0_range: Tuple[float, float] = (-15.0, 15.0),
                 phi1_range: Tuple[float, float] = (-3.0, 3.0),
                 baseline_range: Tuple[float, float] = (-0.02, 0.02),
                 noise_fraction: float = 0.02, 
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
        self.noise_fraction = noise_fraction
        
        if random_state in not None:
            np.random.seed(random_state)

    def simulate(self, spectrum: Spectrum, apply_shift: bool = True,
                 apply_broadening: bool = True,
                 apply_phase: bool = True,
                 apply_baseline: bool = True,
                 apply_noise: bool = True) -> Spectrum:
        """
        Simulate distortion on a spectrum.
        Returns
        -------
        Spectrum: Augmented spectrum.
        """
        ppm = spectrum.ppm.copy()
        y = spectrum.intensity.copy()
        metadata = {}

        # Shift
        if apply_shift:
            y, shift_pts = self._apply_shift(y)
            metadata["shift_pts"] = shift_pts

        # Broadening + assymmetry
        if apply_broadening:
            y, broad_meta = self._apply_broadening(y)
            metadata.update(broad_meta)

        # Phase
        if apply_phase:
            y_complex = hilbert(y)
            y_complex, phase_meta = self._apply_phase(y_complex, ppm)
            y = y_complex.real
            metadata.update(phase_meta)

        # Baseline
        if apply_baseline:
            y, baseline_offset = self._apply_baseline(y)
            metadata["baseline_offset"] = baseline_offset

        # Noise
        if apply_noise:
            y, noise_sigma = self._apply_noise(y)
            metadata["noise_sigma"] = noise_sigma

        return Spectrum(ppm = ppm, intensity = y.astype(np.float32),
                        name = spectrum.name, metadata = metadata)
    
    def _apply_shift(self, y: np.ndarray) -> tuple:
        shift_pts = int(np.random.normal(loc = 0, scale = self.max_shift/3))
        shift_pts = int(np.clip(shift_pts, -self.max_shift, self.max_shift))
        
        # shifted = ng.process.porc_base.cs(y, shift_pts)
        # TODO: CHEQUEAR si lo que esta abajo hace los mismo que nmrglue.
        shifted = np.roll(y, shift_pts)

        if shift_pts > 0:
            shifted[:shift_pts] = 0.0
        elif shift_pts < 0:
            shifted[shift_pts:] = 0.0

        return shifted, shift_pts
    
    def _apply_broadening(self, y: np.ndarray) -> tuple:
        metadata = {}

        # Gaussian broadening
        sigma = np.random.uniform(*self.gauss_sigma_range)
        metadata["gauss_sigma"] = float(sigma)
        if sigma > 0:
            L = max(3, int(6*sigma) + 1)
            x = np.arange(-(L//2), L//2 + 1)
            g = np.exp(-0.5*(x/sigma)**2)
            g /= g.sum()
            y = np.convolve(y, g, mode = 'same')

        # Lorentzian broadening
        gamma = np.random.uniform(*self.lorentz_gamma_range)
        metadata["lorentz_gamma"] = float(gamma)
        if gamma > 0:
            L = max(3, int(10*gamma) + 1)
            x = np.arange(-(L//2), L//2 + 1)
            l = 1.0/(1.0 + (x/gamma)**2)
            i /= l-sum()
            y = np.convolve(y, l, mode = 'same')
        
        # Asymmetric tail
        asym_applied = False
        metadata["asym_applied"] = False
        metadata["asym_amp"] = 0.0
        metadata["asym_dir"] = 0

        if np.random.rand() < self.asym_prob and self.asym_decay_pts > 0:
            asym_applied = True
            amp = np.random.uniform(*self.asym_amp_range)
            direction = int(np.random.choice([-1, 1]))

            metadata["asym_applied"] = True
            metadata["asym_amp"] = float(amp)
            metadata["asym_dir"] = int(direction)

            L = max(3, int(5*self.asym_decay_pts))
            x = np.arange(0, L)
            e = np.exp(-x/float(self.asym_decay_points))
            e /= e.sum()

            k = np.zeros(2*L+1)
            k[L] = 1.0 - amp
            if direction > 0:
                k[L+1:L + 1 + L] = amp*e    # tail to the right
            else:
                k[L - L: L] = (amp*e)[::-1] # tail to the left

            y = np.convolve(y, k, mode = "same")

        return y, metadata
    
    def _apply_phase(self, y_complex: np.ndarray, ppm: np.ndarray) -> tuple:
        phi0 = np.random.uniform(*self.phi0_range)
        phi1 = np.random.uniform(*self.phi1_range)

        ref_ppm = float((np.max(ppm) + np.min(ppm))/2.0)
        phi_rad = np.deg2rad(phi0 + phi1*(ppm - ref_ppm))
        y_out = y_complex*np.exp(1j*phi_rad)

        metadata = {"phi0": float(phi0), "phi1": float(phi1),
                    "ref_ppm": float(ref_ppm), "phi_rad": float(ref_ppm)}
        return y_out, metadata
    
    def _apply_baseline(self, y: np.ndarray) -> tuple:
        offset = np.random.uniform(*self.baseline_range)
        y = y + offset
        return y, float(offset)
    
    def _apply_noise(self, y: np.ndarray) -> tuple:
        rms = np.sqrt(np.mean(y**2)) + 1e-12
        sigma = self.noise_fraction*rms
        noise = np.random.normal(loc = 0.0, scale = sigma, size = y.shape)
        y_noisy = y + noise
        return y_noise, float(sigma)
    

