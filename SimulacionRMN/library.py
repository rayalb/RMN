"""
library.py

Spectrum library management.

Responsibilities:
* Load metadata.
* Load individual spectra from disk.
* Cache spectra in memory.
* Optionally preprocess spectra.
* Retrive spectra by compound name.

"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from .spectrum import Spectrum
from .preprocessing import preprocess_spectrum


class SpectrumLibrary:
    """
    Library of individual compound spectra.
    
    Examples
    --------
    lib = SpectrumLibrary(metadata_path = "metadata.xlsx",
                        spectra_dir = "individual_spectra")
                         
    glucose = lob.get("Glucose")
    glucose_pre = lib.get("Glucose", preprocess = True, ppm_min = 0.0,
                            ppm_max = 10)   
    """

    def __init__(self, metadata_path: str | Path, spectra_dir: str | Path):
        self.metadata_path = Path(metadata_path)
        self.spectra_dir = Path(spectra_dir)

        self.metadata: Optional[pd.DataFrame] = None
        self.comp_to_file: dict[str, str] = {}

        self.cache_raw: dict[str, Spectrum] = {}
        self.cache_processed: dict[tuple, Spectrum] = {}

        self.load_metadata()

    def load_metadata(self) -> None:
        """
        Load Excel metada table.
        Expected columns
        ----------------
        nombre_compuesto
        nombre_archivo_csv
        """

        metadata = pd.read_excel(self.metadata_path)
        self.metadata = metadata
        self.comp_to_file = {row["nombre_compuesto"]:
                             row["nombre_archivo_csv"]
                             for _, row in metadata.iterrows()
                            }
    
    def metadata_row(self, compound_name: str) ->pd.Series:
        """
        Retrieves metadata row corresponding to a compound
        """
        if self.metadata is None:
            self.load_metadata()

        rows = self.metadata[self.metadata["nombre_compuesto"] == compound_name]

        if len(rows) == 0:
            raise KeyError(f"Compound '{compound_name}' not found.")
        
        return rows.iloc[0]
    
    def filename(self, compound_name: str) -> str:
        """
        Return associated spectrum filename.
        """

        if compound_name not in self.comp_to_file:
            raise KeyError(f"Compound '{compound_name}' not found.")
        
        return self.comp_to_file[compound_name]
    

    def _load_spectrum_from_disk(self, compound_name: str) -> Spectrum:
        """
        Load a spectrum from disk

                Returns
        -------
        Spectrum
        """

        if compound_name not in self.comp_to_file:
            raise KeyError(f"Compound '{compound_name}' not found.")
        
        filename = self.comp_to_file[compound_name]
        filepath = self.spectra_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(filepath)
        
        df = pd.read_csv(filepath, sep = "\t")

        ppm = df.iloc[:, 0].to_numpy(dtype = float)
        intensity = df.iloc[:, 1].to_numpy(dtype = float)

        return Spectrum(ppm = ppm, intensity = intensity.astype(np.float32),
                        name = compound_name,
                        metadata = {"compound": compound_name,
                                    "filename": filename})
    
    def get_raw(self, compound_name: str) -> Spectrum:
        """
        Retrieve raw spectrum. Uses cache when possible
        """
        if compound_name not in self.cache_raw:
            self.cache_raw[compound_name] = self._load_spectrum_from_disk(compound_name)

        return self.cache_raw[compound_name].copy()
    
    def get(self, compound_name: str, preprocess: bool = False,
            **preprocess_kwargs) -> Spectrum:
        """
        Retrieve spectrum.
        Parameters
        ----------
        compound_name: str
        preprocess: bool. If True applies preprocessing.
        preprocess_kwargs: Forwardd to preprocess_spectrum()
        """

        if not preprocess:
            return self.get_raw(compound_name)
        
        cache_key = compound_name, tuple(sorted(preprocess_kwargs.items()))

        if cache_key not in self.cache_processed:
            raw = self.get_raw(compound_name)
            processed = preprocess_spectrum(raw, **preprocess_spectrum)
            self.cache_processed[cache_key] = processed

        return self.cache_processed[cache_key].copy()
    
    def preload(self, preprocess: bool = False, **preprocess_kwargs) -> None:
        """
        Load all compounds into cache.
        """
        for compound in self.compounds:
            self.get(compound, preprocess = preprocess, **preprocess_kwargs)

    def clear_cache(self) -> None:
        """
        Clear all cached spectra
        """

        self.cache_raw.clear()
        self.cache_processed.clear()
        
    def compounds(self) -> list[str]:
        """
        List all compounds names.
        """
        return list(self.comp_to_file.keys())

    def __len__(self) -> int:

        return len(self.comp_to_file)

    def __contains__(self, compound_name: str) -> bool:

        return compound_name in self.comp_to_file
    
    def __repr__(self) -> str:
        return f"SpectrumLibrary(n_compounds = {len(self)})"
    
    def summary(self) -> None:
        """
        Print library summary
        """
        print("Spectrum Library")
        print("-"*40)

        print(f"Compounds : {len(self)}")
        print(f"Raw cached : {len(self.cache_raw)}")
        print(f"Processed cached : {len(self.cache_processed)}")



        


    