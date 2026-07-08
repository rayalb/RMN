"""
Example:

Load several spectrum and plot them
"""

import os
from pathlib import Path

from SimulacionRMN.library import SpectrumLibrary
from SimulacionRMN.plotting import plot_spectra


root_dir = Path("/home/ray/Documents/Quimica/") #INTI


metadata = os.path.join(root_dir, "DataSet", "metadata_espectros_individuales_con_TSP sodico(SI).xlsx")
spectra = os.path.join(root_dir, "Espectros individuales + TSP sodico (SI)")

# Load Library

library = SpectrumLibrary(metadata_path = metadata, spectra_dir = spectra)

spectral_list = [library.get_raw("glucosa"), library.get_raw("fructosa"),
                 library.get_raw( "sacarosa")]

plot_spectra(spectral_list, linewidth = 1.2, labels = ["Glucose", "Fructose","Sucrose"],
             title = "Sugars")


