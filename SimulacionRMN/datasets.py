"""
datasets.py

Generation of (R, S, y) datasets for compound detection.

Definitions
===========
    R : reference (individual compound detection).
    S : misture spectrum
    y : binary label
        1 -> compound present
        0 -> compound absent
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from mixture import MixtureSimulator


@dataclass
class PairDataset:
    """
    Dataset conatining (R, S, y) pairs.
    
    Attributes
    ----------
        R : np.ndarray, shape (N_pairs, N_points)
        S : np.ndarray, shape (N_pairs, N_points)
        y : np.ndarray, shape (N_pairs, 1)
        compounds : np.ndarray, compound name for each pair.
        mix_ids :  np.ndarray, mixture id for each pair.
        mix_composition : dict, full composition of efery generated mixture.
    """

    R: np.ndarray
    S: np.ndarray
    y: np.ndarray

    compounds: np.ndarray
    mix_ids: np.ndarray

    mix_compostion: Dict

    def __len__(self):
        return len(self.y)
    
    @property
    def n_pairs(self):
        return len(self.y)
    
    @property
    def n_features(self):
        return self.R.shape[1]
    
    @property
    def positive_fraction(self):
        return float(self.y.mean())
    
    def summary(self):
        print("\nDataset Summary")
        print("---------------------")
        print(f"Pairs: {len(self)}")
        print(f"Features: {self.n_features}")
        print(f"Positives: {self.y.sum():.0f}")
        print(f"Negatives: {(1-self.y).sum():.0f}")
        print(F"Positive %: {100*self.y.mean():.2f}")


class PairDatasetBuilder:
    """
    Generates datasets from a compound library and a 
    MixtureSimulator
    """
    def __init__(self, simulator: MixtureSimulator, compound_library):
        self.simulator = simulator
        self.library = compound_library

    # Build one split

    def build_split(self, n_mixtures: int, verbose: bool = True) -> PairDataset:
        """
        Generate a dataset split.
        Parameters
        ----------
            n_mixtures : int, number of mixtures to generate.
            verbose : bool
            
        Returns
        -------
            PairDataset
        """

        R_list, S_list, y_list = [], [], []
        compound_list, mix_id_list = [], []
        mix_composition = {}

        iterator = range(n_mixtures)
        if verbose:
            iterator = tqdm(iterator, desc = "Generating mixture")

        for mix_id in iterator:
            # Generate mixture
            mixture = self.simulator.simulate()
            mix_composition[mix_id] = mixture.composition.copy()

            S = mixture.spectrum.intesity.astype(np.float32)

            # Build (R, S, y) pairs
            for compound_name, spectrum in self.library.compounds.items():
                R = spectrum.intensity.astype(np.float32)
                y = float(compound_name in mixture.composition)

                R_list.append(R)
                S_list.append(S)
                y_list.append(y)

                compound_list.append(compound_name)
                mix_id_list.append(mix_id)

        return PairDataset(R = np.stack(R_list), S = np.stack(S_list),
                           y = np.array(y_list, dtype = np.float32).reshape(-1, 1),
                           compounds = np.array(compound_list), mix_ids = np.array(mix_id_list),
                           mix_compostion = mix_composition)
        
    def train_valid_test_split(self, n_total: int = 2000, train_frac: float = .7,
                               valid_frac: float = .15, verbose: bool = True):
        """
        Build train/validation/test datasets
        """
        n_train = int(n_total*train_frac)
        n_valid = int(n_total*valid_frac)
        n_test = n_total - n_train - n_valid
        
        print(f"\nSplit sizes:"
              f"\n Train = {n_train}"
              f"\n Validation = {n_valid}"
              f"\n Test = {n_test}")
        
        print("\n Build Train Dataset")
        train = self.build_split(n_train, verbose = verbose)

        print("\n Building Validation Dataset")
        valid = self.buildsplit(n_valid, verbose = verbose)

        print("\n Building Test")
        test = self.build_split(n_test, verbose = verbose)

        return train, valid, test
    
    # CHECK THIS!!
    try:
        import torch
        from torch.utils.data import Dataset

        class PairTorchDataset(Dataset):
            """
            Pytorch wrapper araound PairDataset.
            
            Returns
            -------
            R : torch.FloatTensor
            S : torch.FloatTensor
            y : torch.FloatTensor
            """

            def __init__(self, pair_dataset: PairDataset):
                self.ds = pair_dataset

            def __len__(self):
                return len(self.ds)
            
            def __getitem__(self, index):
                return (torch.from_numpy(self.ds.R[index]).float(),
                        torch.from_numpy(self.ds.S[index]).float(),
                        torch.from_numpy(self.ds.y[index]).float())
            
    except ImportError:
        PairTorchDataset = None

# Utility conversion
def to_torch_dataset(dataset: PairDataset):
    """
    Convert PairDataset to Pytorch dataset.
    Raises
    ------
        ImportError, if torch in not installed.
    """

    if PairTorchDataset is None:
        raise ImportError("Pytorch not installed")
    
    return PairTorchDataset(dataset)

if __name__ == "__main__":
    """
    Example usage
    """
    from library import CompoundLibrary
    from mixture import MixtureSimulator
    #from datasets import PairDatasetBuilder
    
    library = CompoundLibrary(...)
    simulator = MixtureSimulator(library = library)
    builder = PairDatasetBuilder(simulator = simulator, compound_library = library)

    train, valid, test = builder.train_valid_test_split(n_total = 5000)

    train.summary()

    # PyTorch
    # from datasets import PAirTorchDataset
    from torch.utils.data import DataLoader

    train_ds = PairTorchDataset(train)

    loader = DataLoader(train_ds, batch_size = 64, shuffle = True)

    for R, S, y in loader:
        print(R.shape)
        print(S.shape)
        print(y.shape)
        break
    