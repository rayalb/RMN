"""
io.py

Input/output utilities for NMR spectra.

Responsibilities
----------------
* Read spectra from disk.
* Save spectra to disk.
* Read metadata tables.
* Read/write serialized datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .spectrum import Spectrum

PathLike = Union[str, Path]

def read_spectrum_csv(path: PathLike, sep: str = "\t", header: Optional[int] = None) -> Spectrum:
    """
    Read a pectrum form CSV/TXT.
    
    Expected format: ppm intensity
    
    Paramaters
    ----------
    path : str | Path
    sep : str 
        Column separator.
        
    Return
    ------
    Spectrum
    """

    path = Path(path)
    df = pd.read_csv(path, sep = sep, header = header)
    ncols = df.shape[1]

    if ncols == 2:
        ppm = df.iloc[:, 0].to_numpy(dtype = float)
        real = df.iloc[:, 1].to_numpy(dtype = float)
        imag = None
    elif ncols >= 3:
        ppm = df.iloc[:, 0].to_numpy(dtype = float)
        real = df.iloc[:, 1].to_numpy(dtype = float)
        imag = df.iloc[:, 2].to_numpy(dtype = float)
    else:
        raise ValueError(f"{path} must contain at least two columns (ppm, intensity)")
    
    return Spectrum(ppm = ppm, real = real, imag = imag,
                    metadata = {"filename": path.name,
                                "filepath": str(path),
                                "has_imag": imag is not None,
                                "n_points": len(ppm)}) 


def save_spectrum_csv(spectrum: Spectrum, path: PathLike):
    """
    Save spectrum to csv
    """
    path = Path(path)

    if spectrum.imag is None:
        df = pd.DataFrame({"ppm": spectrum.ppm,
                           "real": spectrum.real})
    else:
        df = pd.DataFrame({"ppm": spectrum.ppm,
                           "real": spectrum.real,
                           "imag": spectrum.imag})
    
    
    df.to_csv(path, index = False)


def read_metadata_excel(path: PathLike) -> pd.DataFrame:
    """
    Read metadata Excel file
    """
    return pd.read_excel(path)

def read_metadata_csv(path: PathLike) -> pd.DataFrame:
    """
    Read metadata CSV file
    """
    return pd.read_csv(path)

def list_spectra(directory: PathLike, suffixes = (".csv", ".txt")):
    """
    List spectrum files in a directory.
    """
    directory = Path(directory)
    files = []
    for suffix in suffixes:
        files.extend(directory.glob(f"*{suffix}"))

    return sorted(files)

def load_spectrum_directory(directory: PathLike, sep = "\t"):
    """
    Load all spectra in a folder.
    Returns
    -------
        dict[str, Spectrum]
    """
    spectra = {}
    for path in list_spectra(directory):
        spectra[path.stem] = read_spectrum_csv(path, sep = sep)

    return spectra


def save_dataset_npz(dataset: dict, path: PathLike):
    """
    Save generated dataset.
    
    Example: save_dataset_npz(train, "train.npz)
    """
    np.savez_compressed(path, **dataset)

def load_dataset_npz(path: PathLike) -> dict:
    """
    Load saved dataset.
    """
    data = np.load(path, allow_pickle = True)

    return {kk: data[kk] for kk in data.files}

def build_compound_mapping(metadata: pd.DataFrame, compound_col: str = "nombre_compuesto",
                           filename_col: str = "nombre_archivo_csv") -> dict:
    """
    Build
        compuond -> filename
    mapping from metada.
    """
    return {
            row[compound_col]: row[filename_col] for _, row in metadata.iterrows()
            }
    