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

from spectrum import Spectrum

PathLike = Union[str, Path]

def read_spectrum_csv(path: PathLike, sep: str = "\t") -> Spectrum:
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
    df = pd.read_csv(path = path, sep = sep)

    if df.shape[1] < 2:
        raise ValueError(
            f"{path} must contain at least two columns (ppm, intensity)"
        )
    ppm = df.iloc[:, 0].to_numpy(dtype = float)
    intensity = df.iloc[:, 1]. to_numpy(dtype = float)

    return Spectrum(ppm = ppm, intensity = intensity,
                    metadata = {"filename": path.name,
                                "filepath": str(path)})

def read_complex_spectrum_csv(path: PathLike, sep: str = "\t")->Spectrum:
    """
    Read spectrum with columns: ppm | real | imaginary
    """
    path = Path(path)
    df = pd.read_csv(path, sep = sep)
    if df.shape[1] < 3:
        raise ValueError(
            "Complex spectrum required ppm, real, imag columns."
        )
    ppm = df.iloc[:, 0].to_numpy(dtype = float)
    real = df.iloc[:, 1].to_numpy(dtype = float)
    imag = df.iloc[:, 2].to_numpy(dtype = float)

    intensity = real + 1j*imag

    return Spectrum(ppm = ppm, intensity = intensity,
                    metadata = {"filename": path.name,
                                "filepath": str(path),
                                "complex": True})

def save_spectrum_csv(spectrum: Spectrum, path: PathLike):
    """
    Save spectrum to csv
    """
    path = Path(path)
    df = pd.DataFrame(
        {"ppm": spectrum.ppm,
         "intensity": spectrum.intesity
        })
    df.to_csv(path, index = False)

def save_complex_spectrum_csv(spectrum: Spectrum, path: PathLike):
    """
    Save complex spectrum
    Output:
    ppm | real | imag
    """
    path = Path(path)
    intensity = np.asarray(spectrum.intensity)
    df = pd.DataFrame(
        {"ppm": spectrum.ppm,
         "real": intensity.real,
         "imag": intensity.imag} 
    )
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
        files.extend(directory.glob(f"{suffix}"))

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
    