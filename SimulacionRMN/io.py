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


import os 
import unicodedata
import warnings

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .spectrum import Spectrum

PathLike = Union[str, Path]


def normalize_name(name: str) -> str:
    """
    Normalize filenames/compound names for consistent matching.
    Removes:
        - directory path
        - extension
        - accents   
        - upper/lower case differences
    """
    name = os.path.basename(str(name))
    name = os.path.splitext(name)[0]
    name = name.lower()
    name =  unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    return name.strip().replace(" ", "_")

def to_float_series(series: pd.Series) -> pd.Series:
    """
    Convert a strings to floats, supporting comma decimals.
    Invalid values are converted to NaN.
    """
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex = False), 
                         errors = "coerce")

def read_spectrum_file(path: PathLike, sep: str | None = None) -> Spectrum:
    """
    Read spectrum file.
    
    Expected format: ppm real or ppm real imag
    
    Paramaters
    ----------
    path : str | Path
    sep : str, optional 
        Column separator. If None, pandas will automatically detects it.
        
    Return
    ------
    Spectrum
    """

    path = Path(path)

    kwargs = dict(engine = "python")
    if sep is not None:
        kwargs["sep"] = sep
    else:
        kwargs["sep"] = None

    df = pd.read_csv(path, **kwargs)
    # If first column is not numeric, pandas probably read a header correctly.
    # Otherwise we reload treating the first line as data.
    try:
        float(str(df.columns[0]))
        has_header = False
    except Exception:
        has_header = True

    if not has_header:
        df = pd.read_csv(path, **kwargs, header = None)

    ncols = df.shape[1]
    if ncols < 2:
        raise ValueError(f"{path} must contain at least two columns (ppm, intensity)")

    if ncols == 2:
        df.columns = ["ppm", "real"]
        df["imag"] = None
        has_imag = False
    else:
        df = df.iloc[:, :3]
        df.columns = ["ppm", "real", "imag"]
        has_imag = True

    df["ppm"] = to_float_series(df["ppm"])
    df["real"] = to_float_series(df["real"])
    df["imag"] = to_float_series(df["imag"])

    df = df.dropna(subset = ["ppm", "real"])
    
    ''' 
    if sep is None:
        df = pd.read_csv(path, sep = None, engine = "python", header = None)
    else:
        df = pd.read_csv(path, sep = sep, header = None)
    
    ncols = df.shape[1]

    if ncols < 2:
        raise ValueError(f"{path} must contain at least two columns (ppm, intensity)")

    ppm = df.iloc[:, 0].to_numpy(dtype = float)
    real = df.iloc[:, 1].to_numpy(dtype = float)
    imag = None
    if ncols >= 3:
        imag = df.iloc[:, 2].to_numpy(dtype = float)
    
    mask = np.isfinite(ppm) & np.isfinite(real)
    if imag is not None:
        mask &= np.isfinite(imag)

    ppm = ppm[mask]
    real = real[mask]
    if imag is not None:
        imag = imag[mask]
    '''
    return Spectrum(ppm = df["ppm"].to_numpy(float), real = df["real"].to_numpy(float), 
                    imag = df["imag"].to_numpy(float) if df["imag"].notnull().any() else None,
                    name = path.stem,
                    metadata = {"filename": path.name,
                                "filepath": str(path),
                                "has_imag": has_imag,
                                "n_points": len(df)}) 

def write_spectrum_csv(spectrum: Spectrum, path: PathLike, sep: str = "\t"):
    """
    Save spectrum to csv.
    If imag is available, three columns are written.
    Otherwise, only two columns are save
    """
    path = Path(path)

    if spectrum.imag is None:
        df = pd.DataFrame({"ppm": spectrum.ppm,
                           "real": spectrum.real})
    else:
        df = pd.DataFrame({"ppm": spectrum.ppm,
                           "real": spectrum.real,
                           "imag": spectrum.imag})
    
    
    df.to_csv(path, sep = sep, index = False, header = False)


def load_individual_spectrum(path: PathLike, sep: str = "\t", suffixes = (".csv", ".txt")) -> Spectrum:
    """
    Load every spectrum in a directory.
    Returns
    -------
        spectra: list[Spectrum]
        metadata: list[dict]
    """
    folder = Path(path)
    spectra = []
    metadata = []
    files = sorted([ff for ff in folder.iterdir() if ff.is_file() and ff.suffix.lower() in suffixes and not ff.name.startswith("~$")])

    for file in files:
        spec = read_spectrum_file(file)
        spectra.append(spec)
        metadata.append({
            "filename": file.name,
            "compound":normalize_name(file.stem),
            "n_points": spec.n_points,
            "has_imag": spec.imag is not None
        })
    return spectra, metadata

def load_metadata(path: PathLike, sheet_name: str = "General") -> pd.DataFrame:
    """
    Load metadata table from Excel.
    """
    return pd.read_excel(path, sheet_name = sheet_name)
    
def match_spectra_to_metada(df: pd.DataFrame, metadata: list[dict]):
    """
    Match metadata table with loaded spectra.
    """
    if "csv" in df.columns:
        filename_col = "csv"
    elif "filename_csv" in df.columns:
        filename_col = "filename_csv"
    elif "filename" in df.columns:
        filename_col = "filename"
    else:
        raise ValueError("Metadata must contain csv or filename column.")
    
    df = df.copy()
    df["filename_norm"] = df[filename_col].apply(normalize_name)
    loaded = pd.DataFrame(metadata)
    loaded["idx"] = np.arange(len(loaded))
    loaded["filename_norm"] = loaded["filename"].apply(normalize_name)

    master = df.merge(loaded[["filename_norm", "idx", "filename"]], on = "filename_norm",
                      how = "left", validate = "m:1")
    
    missing = master[master["idx"].isna()]
    if len(missing):
        warnings.warn(f"{len(missing)} spectra could not be matched.", UserWarning)
                       
    diagnostics = {
        "total": len(master), "matched": len(master) - len(missing), "missing": len(missing),
        "missing": len(missing), "missing_df": missing,
        "filename_col": filename_col, "loaded": loaded, 
    }
    return master, diagnostics

def load_internal_standard(path: PathLike) -> Spectrum:
    """
    Load internal standard spectrum.
    """
    spectra = read_spectrum_file(path)
    spectra.name = "TSP"
    spectra.metadata["role"] = "internal_standard"

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
        from the matched metadata table.
    Parameters
    ----------
        metadata : pd.DataFrame,
                Output of match_spectra_to_metada().
        compound_col : str, 
                Column containing  compounds names.
        filename_col : str, 
                Column containing spectrum filenames
    Returns
    -------
        dict[str, str]
            Mapping from compound names to spectrum filenames.
    """
    mapping = {}
    for _, row in metadata.iterrows():
        if pd.isna(row[filename_col]):
            continue
        mapping[row[compound_col]] = row[filename_col]
    return mapping