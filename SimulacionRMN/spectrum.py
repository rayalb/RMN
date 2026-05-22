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
