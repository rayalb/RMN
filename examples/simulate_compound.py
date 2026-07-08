"""
Example:
Load an individual expectrum and preprocess it
"""

import os
from pathlib import Path

from SimulacionRMN.library import SpectrumLibrary

root_dir = Path("/home/ray/Documents/Quimica/") # INTI

metadata = os.path.join(root_dir, "DataSet", "metadata_espectros_individuales_con_TSP sodico(SI).xlsx")
spectra = os.path.join(root_dir, "Espectros individuales + TSP sodico (SI)")

library = SpectrumLibrary(metadata_path = metadata, spectra_dir = spectra)

glucose = library.get("glucosa", preprocess = True)

glucose.plot()