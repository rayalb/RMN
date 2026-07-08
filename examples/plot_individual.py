"""
Example:

Load an individual spectrum and plot it.
"""
import os 
from pathlib import Path

from SimulacionRMN.library import SpectrumLibrary

# Path

#root_dir = Path("/home/ray/Documents/") # CASA
root_dir = Path("/home/ray/Documents/Quimica/") #INTI


metadata = os.path.join(root_dir, "DataSet", "metadata_espectros_individuales_con_TSP sodico(SI).xlsx")
spectra = os.path.join(root_dir, "Espectros individuales + TSP sodico (SI)")

# Load Library

library = SpectrumLibrary(metadata_path = metadata, spectra_dir = spectra)

print(library)

print("Available compounds: ")
print(library.compounds()[:10])

# Load one spectrum

spec = library.get_raw("glucosa")
print(spec)

print(spec.metadata)

spec.plot(figsize = (10, 4), color = "black",
          linewidth = 1.2)
