"""
Example:

Load an individual spectrum and plot it.
"""

from pathlib import Path

from SimulacionRMN.library import SpectrumLibrary

# Path

metadata = Path("data/metadata_individual.xlsx")
spectra = Path("data/individual_spectra")

# Load Library

library = SpectrumLibrary(metadata_path = metadata)
spectra_dir = spectra

library.load_metadata()

print(library)

print("Available compounds: ")
print(library.compounds()[:10])

# Load one spectrum

spec = library.get_raw("Glucose")
print(spec)

print(spec.metadata)

spec.plot(figsize = (10, 4), color = "black",
          linewidth = 1.2)
