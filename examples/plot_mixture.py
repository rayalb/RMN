"""
Example
Load several spetrums and mix them
"""

import os
from pathlib import Path

from SimulacionRMN.library import SpectrumLibrary
from SimulacionRMN.mixture import MixtureSimulator
from SimulacionRMN.augmentation import SpectrumSimulator

from SimulacionRMN.plotting import plot_mixture, plot_mixture_components


#root_dir = Path("/home/ray/Documents/RMN/") # CASA
root_dir = Path("/home/ray/Documents/Quimica/") # INTI

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


# Augment after building the mixture

sim_random_mixtures_2 = MixtureSimulator(library)
mixture_random = sim_random_mixtures_2.simulate_random(n_compounds = 4)
augment = SpectrumSimulator(max_shift = 50, noise_fraction=0.1)
mixture_aug = augment.simulate(mixture_random)

mixture_random.plot(title = 'Original Mixture')
mixture_aug.plot(title = 'Augmented Mixture')

# Hybrid: 1) augment the compounds with chemical variability.
#         2) Augment the final mixture with instrument variability.

simulator = MixtureSimulator(library)
compound_aug = SpectrumSimulator(max_shift = 0, noise_fraction = 0.0, baseline_range = (0, 0))
instrument_aug = SpectrumSimulator(max_shift = 50, noise_fraction = 0.01)

glucose = compound_aug.simulate(library.get("glucosa"))
acido_lactico = compound_aug.simulate(library.get("acido lactico"))
fructose = compound_aug.simulate(library.get("fructosa"))

mixture = simulator.simulate_from_dict(spectra = [glucose, fructose, acido_lactico])

mixture = instrument_aug.simulate(mixture)

