"""
Example:

Load an individual spectrum and plot it.
"""

from pathlib import Path

from SimulacionRMN.library import SpectrumLibrary

# Path

root_dir = Path("/home/Documents/")

metadata = (root_dir, "RMN", "metadata_espectros_individuales_con_TSP sodico(SI).xlsx")
spectra = (root_dir, "RMN", "Espectros individuales + TSP sodico (SI)")

# Load Library

library = SpectrumLibrary(metadata_path = metadata, spectra_dir = spectra)

print(library)

print("Available compounds: ")
print(library.compounds()[:10])

# Load one spectrum

spec = library.get_raw("Glucose")
print(spec)

print(spec.metadata)

spec.plot(figsize = (10, 4), color = "black",
          linewidth = 1.2)
