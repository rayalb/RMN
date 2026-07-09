"""
Example
Load several spetrums and mix them
"""

import os
from pathlib import Path

from SimulacionRMN.library import SpectrumLibrary
from SimulacionRMN.mixture import MixtureSimulator
from SimulacionRMN.plotting import plot_mixture, plot_mixture_components


root_dir = Path("/home/ray/Documents/RMN/") # CASA
#root_dir = Path("/home/ray/Documents/Quimica/") # INTI

metadata = os.path.join(root_dir, "DataSet", "metadata_espectros_individuales_con_TSP sodico(SI).xlsx")
spectra = os.path.join(root_dir, "Espectros individuales + TSP sodico (SI)")

library = SpectrumLibrary(metadata_path = metadata, spectra_dir = spectra)

simulator = MixtureSimulator(library)

mixture = simulator.simulate(["glucosa", "fructosa", "sacarosa"], 
                            concentrations = [0.5, 0.3, 0.2], 
                            normalize_concentrations = True)

print(mixture)
print(mixture.composition)
plot_mixture(mixture)


# Random mixtures

sim_random_mixtures = MixtureSimulator(library, random_state = 2)
mixtures = sim_random_mixtures.simulate_random(n_compounds = 4, normalize_concentrations = False,
                                               store_components = True)

print(mixtures.composition)
plot_mixture(mixtures)

plot_mixture_components(mixtures)