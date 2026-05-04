# generador_espectros.py
# Pipeline for generating synthetic NMR mixture spectra.


# IMPORTS
import os
import re
import random
import unicodedata
import warnings

import numpy as np
import pandas as pd
import nmrglue as ng
from scipy.interpolate import interp1d
from scipy.signal import hilbert


# SECTION 1 — Helpers: loading individual spectra

def _to_float_series(s: pd.Series) -> pd.Series:
    """
    Converts a pandas Series to numeric float values. Replaces commas with dots (European decimal format) and coerces invalid entries to NaN.

    Parameters
        s : pd.Series        Input series with numeric or string values.
    Returns
       pd.Series (float)        Series with valid float values; invalid entries become NaN.
    Used by read_spectrum_file()
    """
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def read_spectrum_file(path: str) -> tuple:
    """
    Reads a spectrum file (tab- or comma-separated) and extracts ppm, real, and imaginary signal arrays. If no imaginary column is present, it is filled with zeros and has_imag is set to False.
    Parameters
    path : str        Full path to the spectrum file (.csv or similar).
    Returns
     ppm      : np.ndarray (float, shape=(N,))       Chemical shift axis in ppm.
    real     : np.ndarray (float, shape=(N,))        Real part of the NMR signal.
    imag     : np.ndarray (float, shape=(N,))        Imaginary part of the NMR signal (zeros if not present).
    has_imag : bool                                  True if the file contained an imaginary column.

    Raises ValueError If the file has an unexpected number of columns or no spectrum found.

    Used by load_individual_spectra(), load_internal_standard()
    """
    df = pd.read_csv(path, sep="\t", header=None, engine="python")

    if df.shape[1] in (2, 3):
        if df.shape[1] == 2:
            df.columns = ["ppm", "real"]
            df["imag"] = 0.0
            has_imag = False
        else:
            df.columns = ["ppm", "real", "imag"]
            has_imag = True

        df["ppm"]  = _to_float_series(df["ppm"])
        df["real"] = _to_float_series(df["real"])
        df["imag"] = _to_float_series(df["imag"])
        df = df.dropna(subset=["ppm", "real"])
        return df["ppm"].to_numpy(float), df["real"].to_numpy(float), df["imag"].to_numpy(float), has_imag

    if df.shape[1] == 1:
        s = df.iloc[:, 0].astype(str)
        parts = ( s.str.split(",", n=2, expand=True) if s.str.contains(",", regex=False).any() else s.str.split(r"\s+", n=2, expand=True) )
        if parts.shape[1] < 2:
            raise ValueError(f"{os.path.basename(path)}: only one column detected — no spectrum found.")

        ppm  = _to_float_series(parts.iloc[:, 0])
        real = _to_float_series(parts.iloc[:, 1])

        if parts.shape[1] >= 3:
            imag     = _to_float_series(parts.iloc[:, 2]).fillna(0.0)
            has_imag = True
        else:
            imag     = pd.Series(np.zeros(len(ppm)))
            has_imag = False

        df2 = pd.DataFrame({"ppm": ppm, "real": real, "imag": imag}).dropna(subset=["ppm", "real"])
        return df2["ppm"].to_numpy(float), df2["real"].to_numpy(float), df2["imag"].to_numpy(float), has_imag

    raise ValueError(f"{os.path.basename(path)}: unexpected number of columns ({df.shape[1]}).")


def resample_to_ref(
    ppm_i_desc  : np.ndarray,
    y_i_desc    : np.ndarray,
    ppm_ref_desc: np.ndarray
) -> np.ndarray:
    """
    Interpolates a spectrum onto a reference ppm axis. Handles the descending NMR convention (10 → 0 ppm) by internally reversing arrays before interpolation and restoring the original order afterward.

    Parameters
        ppm_i_desc   : np.ndarray (float, shape=(N,))        ppm axis of the spectrum to resample (descending order).
        y_i_desc     : np.ndarray (float, shape=(N,))        Signal intensities corresponding to ppm_i_desc.
        ppm_ref_desc : np.ndarray (float, shape=(M,))        Reference ppm axis onto which to interpolate (descending order).
    Returns
        np.ndarray (float, shape=(M,))        Resampled signal on the reference ppm axis (descending order). Points outside the original range are filled with 0.0.
    Used by load_individual_spectra()
    """
    x_i_inc   = ppm_i_desc[::-1]
    y_i_inc   = y_i_desc[::-1]
    x_ref_inc = ppm_ref_desc[::-1]

    f = interp1d( x_i_inc, y_i_inc, kind="linear",bounds_error=False,fill_value=0.0, assume_sorted=True )
    return f(x_ref_inc)[::-1]

def load_individual_spectra(
    root_dir : str,
    subfolder: str   = os.path.join("Moleculas mol", "Espectros individuales"),
    ext      : str   = ".csv",
    atol     : float = 1e-6
) -> tuple:
    """
    Loads all spectrum files from a folder, aligns them to a common ppm reference axis via interpolation, and returns the aligned spectra. The spectrum with the most points is chosen as the reference axis.

    Parameters
   
        root_dir  : str        Base directory path.
        subfolder : str        Relative path from root_dir to the spectra folder. Default: "Moleculas mol/Espectros individuales".
         ext       : str        File extension to filter. Default: ".csv".
        atol      : float      Absolute tolerance for ppm axis comparison. Default: 1e-6.

    Returns
    
        standardsSpec       : list of np.ndarray (float, shape=(N,2))        List of aligned spectra, each as [ppm, real] columns.
        standardsDictionary : list of dict       
            Metadata per spectrum:
                 - filename       : str
                 - points_original: int
                 - has_imag       : bool
                 - resampled      : bool
         x_ref               : np.ndarray (float, shape=(N,))               Reference ppm axis used for alignment (descending order).

    Raises ValueError If no spectrum files are found in the folder.
    Used by  pipeline entry point
    """
    folder = os.path.join(root_dir, subfolder)

    filenames = sorted([ f for f in os.listdir(folder) if f.lower().endswith(ext) and not f.startswith("~$") ])

    standardsSpec       = []
    standardsDictionary = []

    for file in filenames:
        ppm, real, imag, has_imag = read_spectrum_file(os.path.join(folder, file))
        standardsSpec.append(np.column_stack((ppm, real)))
        standardsDictionary.append({ "filename" : file,  "points_original": len(ppm),  "has_imag": has_imag, "resampled"  : False })

    if len(standardsSpec) == 0:
        raise ValueError(f"No spectrum files found in: {folder}")

    # Reference axis = spectrum with the most points
    idx_ref = int(np.argmax([spec.shape[0] for spec in standardsSpec]))
    x_ref   = standardsSpec[idx_ref][:, 0]

    # Resample all spectra that differ from the reference axis
    for i in range(len(standardsSpec)):
        x_i = standardsSpec[i][:, 0]
        y_i = standardsSpec[i][:, 1]

        need_resample = ( len(x_i) != len(x_ref)  or not np.allclose(x_i, x_ref, rtol=0, atol=atol)  )

        if need_resample:
            y_res            = resample_to_ref(x_i, y_i, x_ref)
            standardsSpec[i] = np.column_stack((x_ref, y_res))
            standardsDictionary[i]["resampled"] = True

    return standardsSpec, standardsDictionary, x_ref

# 
# SECTION 2 — Loading metadata and internal standard
# 

def load_metabolic_profile( root_dir  : str, filename  : str = "perfil metabolico vino.xlsx", sheet_name: str = "General") -> pd.DataFrame:
    """
    Loads the metabolic profile table from an Excel file.

    Parameters
         root_dir   : str        Base directory path where the Excel file is located.
        filename   : str        Excel filename. Default: "perfil metabolico vino.xlsx".
        sheet_name : str        Sheet name to read. Default: "General".

    Returns
         df : pd.DataFrame        Raw metabolic profile table as loaded from the Excel sheet.

    Used by  match_spectra_to_metadata()
    """
    excel_path = os.path.join(root_dir, filename)
    return pd.read_excel(excel_path, sheet_name=sheet_name)


def normalize_name(s: str) -> str:
    """
    Normalizes a filename string for robust matching: removes path, extension, converts to lowercase, and strips accents/special characters.

    Parameters
      s : str        Raw filename or path string.

    Returns
        str         Normalized string (lowercase, no extension, no accents).

    Used by match_spectra_to_metadata()
    """
    s = os.path.basename(str(s))
    s = os.path.splitext(s)[0]
    s = s.lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.strip()


def match_spectra_to_metadata( df: pd.DataFrame, standardsDictionary: list,) -> tuple:
    """
    Matches loaded spectra (from standardsDictionary) to their metadata rows (from df) using normalized filenames as keys. Emits warnings instead of printing, and returns a diagnostics dict.

    Parameters
    
    df                  : pd.DataFrame        Metadata table. Must contain one of these columns (in priority order): 'csv', 'filename_csv', or 'filename'.
    standardsDictionary : list of dict        Output of load_individual_spectra(). Each dict must have a 'filename' key.

    Returns
  
    master_df   : pd.DataFrame               df merged with spectrum index and disk filename.
        Added columns:
            - filename_norm : str   (normalized key used for matching)
            - idx           : float (index into standardsSpec; NaN if unmatched)
            - filename_disk : str   (original filename on disk; NaN if unmatched)
    diagnostics : dict
        {
            "total"       : int            — total rows in master_df,
            "matched"     : int            — rows with a valid spectrum match,
            "missing"     : int            — rows without a match,
            "missing_df"  : pd.DataFrame   — subset of unmatched rows,
            "filename_col": str            — column used as filename key in df,
            "df_loaded"   : pd.DataFrame   — full index table (idx, filename_disk, filename_norm)
        }

    Used by _generate_single_mixture()
    """
    # 1) Select filename column from metadata df
    if "csv" in df.columns:
        filename_col = "csv"
    elif "filename_csv" in df.columns:
        filename_col = "filename_csv"
    else:
        filename_col = "filename"

    # 2) Normalize names in metadata df
    df = df.copy()
    df["filename_norm"] = df[filename_col].apply(normalize_name)

    # 3) Build index table from standardsDictionary (preserves order of standardsSpec)
    df_loaded = pd.DataFrame({ "filename_disk": [d["filename"] for d in standardsDictionary], "idx"  : np.arange(len(standardsDictionary)) })
    df_loaded["filename_norm"] = df_loaded["filename_disk"].apply(normalize_name)

    # 4) Merge by normalized key
    master_df = df.merge(df_loaded[["filename_norm", "idx", "filename_disk"]], on="filename_norm", how="left", validate="m:1" )

    # 5) Diagnostics
    missing   = master_df[master_df["idx"].isna()]
    n_total   = len(master_df)
    n_matched = n_total - len(missing)
    n_missing = len(missing)

    if n_missing > 0:
        warnings.warn( f"{n_missing} row(s) in metadata could not be matched to any spectrum file.\n" f"Unmatched entries:\n{missing[[filename_col, 'filename_norm']].head().to_string()}", UserWarning, stacklevel=2 )

    diagnostics = {
        "total"       : n_total,
        "matched"     : n_matched,
        "missing"     : n_missing,
        "missing_df"  : missing[[filename_col, "filename_norm"]].copy(),
        "filename_col": filename_col,
        "df_loaded"   : df_loaded.sort_values("idx").reset_index(drop=True)
    }

    return master_df, diagnostics


def load_internal_standard( root_dir : str,  filename : str = "tsp-d4 sodico.csv",  subfolder: str = "Moleculas mol") -> tuple:
    """
    Loads the internal standard spectrum from a CSV file. Delegates file reading and parsing to read_spectrum_file(),which handles tab/comma separated formats and 1-3 column layouts.
      Parameters
    
    root_dir  : str        Base directory path.
    filename  : str        Filename of the internal standard spectrum. Default: "tsp-d4 sodico.csv".
    subfolder : str        Subfolder inside root_dir where the file is located. Default: "Moleculas mol".
    Returns
     ppm_std : np.ndarray (float, shape=(N,))        Chemical shift axis of the internal standard (ppm).
    v_std   : np.ndarray (float, shape=(N,))        Real signal intensities of the internal standard.

    Used by _generate_single_mixture()
    """
    path_std = os.path.join(root_dir, subfolder, filename)

    ppm_std, v_std, _, _ = read_spectrum_file(path_std)

    # Extra safety: remove any remaining non-finite values
    mask    = np.isfinite(ppm_std) & np.isfinite(v_std)
    ppm_std = ppm_std[mask]
    v_std   = v_std[mask]

    return ppm_std, v_std


# SECTION 3 — Mixture sampling helpers

def entra_en_mezcla(p_ausencia: float) -> bool:
    """
    Decides randomly whether a compound is included in a mixture, based on its absence probability from the metabolic profile.

    Parameters

    p_ausencia : float        Probability of absence of the compound [0.0 - 1.0]. Example: 0.3 means 30% chance of being absent from the mixture.

    Returns
        True  -> compound is included in the mixture.
        False -> compound is absent from the mixture.

    Used by _generate_single_mixture()
    """
    return np.random.rand() > float(p_ausencia)


def sample_conc(media: float, desvio: float, distribucion: str) -> float:
    """
    Samples a compound concentration according to a statistical distribution defined in the metabolic profile. If the distribution string is not recognized, returns the mean value as fallback.

    Supported distributions
        - "normal"      : Gaussian, clipped at 0.
        - "lognormal"   : Log-normal parameterized by mean and std.
        - "gamma"       : Gamma parameterized by mean and std.
        - "rectangular" : Uniform distribution with matching mean and std.
        - "triangular"  : Triangular distribution with matching mean and std.

    Parameters
      media        : float        Mean concentration of the compound.
     desvio       : float        Standard deviation of the concentration.
        distribucion : str        Distribution name (case-insensitive, hyphens/spaces ignored).
      
    Returns
    float        Sampled concentration value (always >= 0.0).

    Used by  _generate_single_mixture()
    """
    m, s = float(media), float(desvio)

    dist     = str(distribucion).strip().lower()
    dist_key = re.sub(r"[\s\-]", "", dist)

    if dist.startswith("normal"):
        return max(0.0, np.random.normal(m, s))

    if dist_key.startswith("lognormal"):
        if m <= 0 or s <= 0:
            return max(0.0, m)
        var    = s**2
        sigma2 = np.log(1.0 + var / (m**2))
        sigma  = np.sqrt(max(1e-12, sigma2))
        mu     = np.log(m) - 0.5 * sigma2
        return max(0.0, float(np.random.lognormal(mean=mu, sigma=sigma)))

    if dist_key.startswith("gamma"):
        if m <= 0 or s <= 0:
            return max(0.0, m)
        k     = (m / s)**2           # shape
        theta = (s**2) / m           # scale
        return max(0.0, float(np.random.gamma(shape=max(1e-12, k), scale=max(1e-12, theta))))

    if dist.startswith("rectangular"):
        w    = s * np.sqrt(12.0)
        a, b = max(0.0, m - w / 2), max(m + w / 2, m - w / 2 + 1e-12)
        return float(np.random.uniform(a, b))

    if dist.startswith("triangular"):
        h    = s * np.sqrt(6.0)
        a, c = max(0.0, m - h), max(m + h, m - h + 1e-12)
        return float(np.random.triangular(a, m, c))

    # Fallback: unrecognized distribution -> return mean
    return max(0.0, m)


# SECTION 4 — Spectral distortion helpers

def corrimiento_aleatorio(spec: np.ndarray, max_shift: int) -> tuple:
    """
    Applies a random shift (in points) to a spectrum along the ppm axis. The shift is sampled from a normal distribution clipped to [-max_shift, max_shift], using max_shift/3 as the standard deviation (so ~99% of shifts stay within bounds).

    Parameters
        spec      : np.ndarray (float, shape=(N,))        Input spectrum signal (real intensities, 1D).
        max_shift : int                                   Maximum allowed shift in points (inclusive).

    Returns

    spec_shifted : np.ndarray (float, shape=(N,))        Shifted spectrum signal.
    shift_pts    : int                                   Actual shift applied in points. Positive = right, negative = left.

    Used by _generate_single_mixture()
    """
    shift_pts    = int(np.random.normal(0, max_shift / 3))
    shift_pts    = int(np.clip(shift_pts, -max_shift, max_shift))
    spec_shifted = ng.process.proc_base.cs(spec, shift_pts)
    return spec_shifted, shift_pts


def deformar_picos(
    spec               : np.ndarray,
    gauss_sigma_range  : tuple = (0.0, 2.0),
    lorentz_gamma_range: tuple = (0.0, 2.0),
    asym_prob          : float = 0.2,
    asym_decay_pts     : int   = 30,
    asym_amp_range     : tuple = (0.0, 0.10)
) -> tuple:
    """
    Applies random peak broadening and asymmetry distortions to a spectrum. Three independent distortions are applied sequentially: Gaussian broadening -> Lorentzian broadening -> asymmetric tail.

    Distortions
    
         - Gaussian  : convolution with a Gaussian kernel (sigma sampled uniformly).
         - Lorentzian: convolution with a Lorentzian kernel (gamma sampled uniformly).
        - Asymmetry : random exponential tail added to one side of peaks, applied with probability asym_prob.

    Parameters
        spec               : np.ndarray (float, shape=(N,))        Input real spectrum signal (1D).
         gauss_sigma_range  : tuple of (float, float)              Uniform sampling range for Gaussian sigma (points). Default: (0.0, 2.0). If sampled sigma == 0, Gaussian broadening is skipped.
        lorentz_gamma_range: tuple of (float, float)               Uniform sampling range for Lorentzian gamma (points). Default: (0.0, 2.0). If sampled gamma == 0, Lorentzian broadening is skipped.
        asym_prob          : float                                 Probability [0.0 - 1.0] of applying asymmetric tail. Default: 0.2.
        sym_decay_pts     : int                                    Decay length of the exponential tail in points. Default: 30.
        asym_amp_range     : tuple of (float, float)               Uniform sampling range for tail amplitude relative to peak.        Default: (0.0, 0.10).

    Returns
        y             : np.ndarray (float, shape=(N,))        Distorted spectrum signal.
        deform_params : dict
                                                                    Parameters of the distortions applied, for logging:
                                                                        - gauss_sigma   : float  — Gaussian sigma used (points).
                                                                        - lorentz_gamma : float  — Lorentzian gamma used (points).
                                                                        - asym_applied  : int    — 1 if asymmetry was applied, 0 otherwise.
                                                                        - asym_amp      : float  — Tail amplitude applied (0.0 if not applied).
                                                                        - asym_dir      : int    — Tail direction: 1=right, -1=left, 0=none.
                                                                        - asym_decay_pts: int    — Decay length used (points).

    Used by  _generate_single_mixture()
    """
    y = spec.copy()

    # Gaussian broadening
    sigma = np.random.uniform(*gauss_sigma_range)
    if sigma > 0:
        L = max(3, int(6 * sigma) + 1)
        x = np.arange(-(L // 2), L // 2 + 1)
        g = np.exp(-0.5 * (x / sigma) ** 2)
        g /= g.sum()
        y = np.convolve(y, g, mode="same")

    # Lorentzian broadening
    gamma = np.random.uniform(*lorentz_gamma_range)
    if gamma > 0:
        L = max(3, int(10 * gamma) + 1)
        x = np.arange(-(L // 2), L // 2 + 1)
        l = 1.0 / (1.0 + (x / gamma) ** 2)
        l /= l.sum()
        y = np.convolve(y, l, mode="same")

    # Asymmetric tail
    asym_applied = 0
    amp          = 0.0
    dir_sign     = 0

    if np.random.rand() < asym_prob and asym_decay_pts > 0:
        asym_applied = 1
        amp          = np.random.uniform(*asym_amp_range)
        dir_sign     = int(np.random.choice([-1, 1]))

        L = max(3, int(5 * asym_decay_pts))
        x = np.arange(0, L)
        e = np.exp(-x / float(asym_decay_pts))
        e /= e.sum()

        k    = np.zeros(2 * L + 1)
        k[L] = 1.0 - amp
        if dir_sign > 0:
            k[L + 1: L + 1 + L] = amp * e           # tail to the right
        else:
            k[L - L: L]         = (amp * e)[::-1]   # tail to the left

        y = np.convolve(y, k, mode="same")

    deform_params = {
        "gauss_sigma"   : float(sigma),
        "lorentz_gamma" : float(gamma),
        "asym_applied"  : int(asym_applied),
        "asym_amp"      : float(amp),
        "asym_dir"      : int(dir_sign),
        "asym_decay_pts": int(asym_decay_pts)
    }

    return y, deform_params


# SECTION 5 — Phase correction helpers

def _ppm_center(ppm_axis: np.ndarray) -> float:
    """
    Computes the center ppm value of a chemical shift axis. Used as the default ref_ppm in phase correction to apply first-order phase symmetrically around the spectral midpoint.

    Parameters
            ppm_axis : np.ndarray (float, shape=(N,))        Chemical shift axis in ppm.
    Returns
        float                                                Midpoint value: (max(ppm) + min(ppm)) / 2.

    Used by choose_phase_strategy()
    """
    return float((np.max(ppm_axis) + np.min(ppm_axis)) / 2.0)


def _ref_from_max_peak(ppm_axis: np.ndarray, y: np.ndarray) -> float:
    """
    Returns the ppm position of the highest intensity peak in the spectrum. Used as an alternative ref_ppm in phase correction, anchoring the first-order phase at the most dominant peak instead of the spectral center.

    Parameters
           ppm_axis : np.ndarray (float, shape=(N,))        Chemical shift axis in ppm.
        y        : np.ndarray (float, shape=(N,))        Spectrum intensities (real or absolute values).
    Returns
        float                                            ppm value corresponding to the maximum absolute intensity peak.

    Used by choose_phase_strategy()
    """
    idx = int(np.argmax(np.abs(y)))
    return float(ppm_axis[idx])


def _ref_from_prominent_random(
    ppm_axis: np.ndarray,
    y       : np.ndarray,
    q       : float = 0.90
) -> float:
    """
    Returns the ppm position of a randomly chosen prominent peak. Prominent peaks are defined as those with absolute intensity above the q-th quantile. If no candidates are found, falls back to the maximum intensity peak via _ref_from_max_peak().

    Parameters
        ppm_axis : np.ndarray (float, shape=(N,))        Chemical shift axis in ppm.
        y        : np.ndarray (float, shape=(N,))        Spectrum intensities (real or absolute values).
        q        : float                                 Quantile threshold [0.0 - 1.0] to define prominent peaks.  Default: 0.90 (top 10% intensity points are candidates).
    Returns
      float                                              ppm value of a randomly selected prominent peak.

    Used by choose_phase_strategy()
    """
    yabs       = np.abs(y)
    thr        = np.quantile(yabs, q)
    candidates = np.where(yabs >= thr)[0]

    if candidates.size == 0:
        return _ref_from_max_peak(ppm_axis, y)

    idx = int(np.random.choice(candidates))
    return float(ppm_axis[idx])


def choose_phase_strategy(
    ppm_axis         : np.ndarray,
    y                : np.ndarray,
    REF_REFERENCE_PPM: float = None,
    PCT_NO_PHASE     : float = 0.30,
    PCT_MAX_PEAK     : float = 0.20,
    PCT_REF_PEAK     : float = 0.15,
    PCT_PROM_PEAK    : float = 0.20,
    PCT_CENTER       : float = 0.15
) -> tuple:
    """
    Randomly selects a phase correction strategy for a spectrum, returning whether to apply phase correction, the reference ppm pivot, and a tag identifying the strategy chosen.

    Strategies (selected by weighted random draw)
        - "no_phase"              : no phase correction applied.
        - "max_peak"              : pivot at the highest intensity peak.
        - "ref_peak"              : pivot at a fixed reference ppm (e.g. TSP at 0.0). Falls back to spectral center if REF_REFERENCE_PPM is None.
        - "prominent_random"      : pivot at a randomly chosen prominent peak (top 10%).
        - "center"                : pivot at the spectral midpoint.

    Parameters
        ppm_axis          : np.ndarray (float, shape=(N,))        Chemical shift axis in ppm.
        y                 : np.ndarray (float, shape=(N,))        Spectrum intensities (real or complex values).
        REF_REFERENCE_PPM : float or None                         Fixed reference ppm for the "ref_peak" strategy (e.g. 0.0 for TSP).  If None, falls back to spectral center. Default: None.
        PCT_NO_PHASE      : float                                 Proportion of cases with no phase correction. Default: 0.30.
        PCT_MAX_PEAK      : float                                 Proportion of cases using max peak as pivot. Default: 0.20.
        PCT_REF_PEAK      : float                                 Proportion of cases using fixed reference ppm. Default: 0.15.
        PCT_PROM_PEAK     : float                                 Proportion of cases using a random prominent peak. Default: 0.20.
        PCT_CENTER        : float                                 Proportion of cases using spectral center as pivot. Default: 0.15.

    Returns
        apply_phase : bool        True if phase correction should be applied, False otherwise.
    ref_ppm     : float or np.nan        Reference ppm pivot for phase correction. np.nan if no phase applied.
    mode_tag    : str        Strategy label for logging: "no_phase" | "max_peak" | "ref_peak" | "ref_peak_fallback_center" |   "prominent_random" | "center".

    Used by _generate_single_mixture()
    """
    cut_no   = PCT_NO_PHASE
    cut_max  = cut_no  + PCT_MAX_PEAK
    cut_ref  = cut_max + PCT_REF_PEAK
    cut_prom = cut_ref + PCT_PROM_PEAK
    # remainder up to 1.0 covers PCT_CENTER

    u = random.random()

    if u < cut_no:
        return False, float("nan"), "no_phase"

    if u < cut_max:
        return True, _ref_from_max_peak(ppm_axis, y), "max_peak"

    if u < cut_ref:
        if REF_REFERENCE_PPM is not None:
            return True, float(REF_REFERENCE_PPM), "ref_peak"
        else:
            return True, _ppm_center(ppm_axis), "ref_peak_fallback_center"

    if u < cut_prom:
        return True, _ref_from_prominent_random(ppm_axis, y), "prominent_random"

    # Fallback: center (covers PCT_CENTER and any floating-point remainder)
    return True, _ppm_center(ppm_axis), "center"


def aplicar_fase(
    y_complex  : np.ndarray,
    x_ppm      : np.ndarray,
    ref_ppm    : float = 0.0,
    prob       : float = 1.0,
    phi0_range : tuple = (-15.0, 15.0),   # degrees
    phi1_range : tuple = (-3.0,   3.0)    # degrees/ppm
) -> tuple:
    """
    Applies a random zero- and first-order phase correction to a complex spectrum. Phase correction is applied with probability `prob`; if skipped, phi0=phi1=0 and the spectrum is returned unchanged.

    Phase model
        phi(ppm) = phi0 + phi1 * (ppm - ref_ppm)     [degrees]
        y_out    = y_complex * exp(i * phi_rad)        [complex rotation]

    Parameters
   
    y_complex  : np.ndarray (complex, shape=(N,))        Input complex spectrum (e.g. real + i*hilbert(real)). Must be complex dtype — raises ValueError otherwise.
    x_ppm      : np.ndarray (float, shape=(N,))          Chemical shift axis in ppm. Must match shape of y_complex.
    ref_ppm    : float                                   Reference ppm point for first-order phase correction. Default: 0.0.
    prob       : float                                   Probability [0.0 - 1.0] of applying phase correction.  Default: 1.0 (always applied).
    phi0_range : tuple of (float, float)                 Uniform sampling range for zero-order phase (degrees). Default: (-15.0, 15.0).
    phi1_range : tuple of (float, float)                 Uniform sampling range for first-order phase (degrees/ppm). Default: (-3.0, 3.0).

    Returns
    y_out   : np.ndarray (complex, shape=(N,))          Phase-corrected complex spectrum.
    aplicar : int                                       1 if phase correction was applied, 0 if skipped.
    phi0    : float                                     Zero-order phase applied (degrees).
    phi1    : float                                     First-order phase applied (degrees/ppm).
    ref_ppm : float                                     Reference ppm used for first-order correction.

    Raises   ValueError  If prob is outside [0, 1], y_complex is not complex, or y_complex and x_ppm have different shapes.

    Used by _generate_single_mixture()
    """
    if not (0.0 <= float(prob) <= 1.0):
        raise ValueError("prob must be between 0 and 1.")
    if not np.iscomplexobj(y_complex):
        raise ValueError("y_complex must be complex dtype (apply hilbert first).")

    y_complex = np.asarray(y_complex)
    x_ppm     = np.asarray(x_ppm, dtype=float)

    if y_complex.shape != x_ppm.shape:
        raise ValueError("y_complex and x_ppm must have the same shape.")

    aplicar = (np.random.rand() < prob)

    phi0    = np.random.uniform(*phi0_range) if aplicar else 0.0
    phi1    = np.random.uniform(*phi1_range) if aplicar else 0.0

    phi_rad = np.deg2rad(phi0 + phi1 * (x_ppm - ref_ppm))
    y_out   = y_complex * np.exp(1j * phi_rad)

    return y_out, int(aplicar), float(phi0), float(phi1), float(ref_ppm)


# SECTION 6 — Mixture generation
# 
def _generate_single_mixture(
    mix_id             : int,
    master_df          : pd.DataFrame,
    standardsSpec      : list,
    ppm_axis           : np.ndarray,
    v                  : np.ndarray,
    max_shift          : int,
    standard_scale     : float = 1.0,
    aplicar_deformacion: bool  = False,
    gauss_sigma_range  : tuple = (0.0, 2.0),
    lorentz_gamma_range: tuple = (0.0, 2.0),
    asym_prob          : float = 0.2,
    asym_decay_pts     : int   = 30,
    asym_amp_range     : tuple = (0.0, 0.10),
    REF_REFERENCE_PPM  : float = None,
    PCT_NO_PHASE       : float = 0.30,
    PCT_MAX_PEAK       : float = 0.20,
    PCT_REF_PEAK       : float = 0.15,
    PCT_PROM_PEAK      : float = 0.20,
    PCT_CENTER         : float = 0.15,
    phi0_range         : tuple = (-15.0, 15.0),
    phi1_range         : tuple = (-3.0,   3.0),
) -> tuple:
    """
    Generates a single synthetic NMR mixture spectrum by:
        1. Randomly including compounds based on absence probability.
        2. Sampling concentrations from their statistical distributions.
        3. Applying random shift and optional peak deformation per compound.
        4. Summing all compound spectra weighted by concentration.
        5. Adding internal standard, baseline offset, and Gaussian noise.
        6. Complexifying via Hilbert transform.
        7. Applying random phase correction.

    Parameters
    
        mix_id              : int                                            Identifier for this mixture (used for logging).
        master_df           : pd.DataFrame        M                          Merged metadata table from match_spectra_to_metadata(). Required columns: 'idx', 'p_ausencia', 'Media', 'Desvío',   'Distribución recomendada', 'Compuesto'.
        standardsSpec       : list of np.ndarray (float, shape=(N,2))        Aligned individual spectra from load_individual_spectra().
        ppm_axis            : np.ndarray (float, shape=(N,))                  Reference ppm axis.
        v                   : np.ndarray (float, shape=(N,))                 Internal standard signal from load_internal_standard().
        max_shift           : int                                            Maximum random shift in points for corrimiento_aleatorio().
        standard_scale      : float                                          Scaling factor for the internal standard signal. Default: 1.0.
        aplicar_deformacion : bool                                           If tue, applies peak deformation via deformar_picos(). Default: False.
        gauss_sigma_range   : tuple of (float, float)                        Passed to deformar_picos(). Default: (0.0, 2.0).
        lorentz_gamma_range : tuple of (float, float)                        Passed to deformar_picos(). Default: (0.0, 2.0).
        asym_prob           : float                                          Passed to deformar_picos(). Default: 0.2.
        asym_decay_pts      : int                                            Passed to deformar_picos(). Default: 30.
        asym_amp_range      : tuple of (float, float)                        Passed to deformar_picos(). Default: (0.0, 0.10).
        REF_REFERENCE_PPM   : float or None                                  Passed to choose_phase_strategy(). Default: None.
        PCT_NO_PHASE        : float                                          Passed to choose_phase_strategy(). Default: 0.30.
        PCT_MAX_PEAK        : float                                          Passed to choose_phase_strategy(). Default: 0.20.
        PCT_REF_PEAK        : float                                          Passed to choose_phase_strategy(). Default: 0.15.
        PCT_PROM_PEAK       : float                                          Passed to choose_phase_strategy(). Default: 0.20.
        PCT_CENTER          : float                                          Passed to choose_phase_strategy(). Default: 0.15.
        phi0_range          : tuple of (float, float)                        Passed to aplicar_fase(). Default: (-15.0, 15.0).
        phi1_range          : tuple of (float, float)                        Passed to aplicar_fase(). Default: (-3.0, 3.0).

    Returns
    
        mix_complex       : np.ndarray (complex, shape=(N,))                    Final synthetic complex spectrum.
        conc_dict         : dict                                                Compound concentrations used: {idx (int): conc (float), '_IS': float}.
        per_compound_rows : list of dict                                        Per-compound parameters for logging (shift, deformation, etc.).
        mix_meta          : dict                                                 Mixture-level metadata for logging:
                                                                                                                        - mix_id          : int
                                                                                                                        - baseline_offset : float
                                                                                                                        - noise_sigma     : float
                                                                                                                        - phase_applied   : int   (0 or 1)
                                                                                                                        - phi0            : float
                                                                                                                        - phi1            : float
                                                                                                                        - ref_ppm         : float or np.nan
                                                                                                                        - phase_mode      : str
    Used by generate_mixture_spectra()
    """
    mix_real          = np.zeros_like(ppm_axis, dtype=float)
    conc_dict         = {}
    per_compound_rows = []

    # 1) Sum compounds
    for _, row in master_df.iterrows():
        if not entra_en_mezcla(row["p_ausencia"]):
            continue

        conc   = sample_conc(row["Media"], row["Desvío"], row["Distribución recomendada"])
        spec_y = standardsSpec[int(row["idx"])][:, 1]

        spec_shifted, shift_pts = corrimiento_aleatorio(spec_y, max_shift)

        if aplicar_deformacion:
            spec_deformed, deform_params = deformar_picos(
                spec_shifted,
                gauss_sigma_range   = gauss_sigma_range,
                lorentz_gamma_range = lorentz_gamma_range,
                asym_prob           = asym_prob,
                asym_decay_pts      = asym_decay_pts,
                asym_amp_range      = asym_amp_range,
            )
        else:
            spec_deformed = spec_shifted.copy()
            deform_params = {}

        mix_real += conc * spec_deformed
        conc_dict[int(row["idx"])] = float(conc)

        row_params = {
            "mix_id"   : mix_id,
            "idx"      : int(row["idx"]),
            "Compuesto": row["Compuesto"],
            "shift_pts": float(shift_pts),
        }
        for dk, val in deform_params.items():
            row_params[f"deform_{dk}"] = float(val)
        per_compound_rows.append(row_params)

    # 2) Add internal standard
    m        = min(len(mix_real), len(v))
    mix_real = mix_real[:m] + standard_scale * v[:m]
    conc_dict["_IS"] = float(standard_scale)

    # 3) Baseline offset
    baseline_offset = float(np.random.uniform(-0.02, 0.02))
    mix_real        = mix_real + baseline_offset

    # 4) Gaussian noise
    rms         = np.sqrt(np.mean(mix_real**2)) + 1e-12
    noise_sigma = float(0.02 * rms)
    mix_real    = mix_real + np.random.normal(0, noise_sigma, size=mix_real.shape)

    # 5) Hilbert -> complex
    mix_complex = hilbert(mix_real)

    # 6) Phase correction strategy
    apply_phase, ref_ppm_sel, mode_tag = choose_phase_strategy(
        ppm_axis[:len(mix_complex)],
        mix_complex,
        REF_REFERENCE_PPM = REF_REFERENCE_PPM,
        PCT_NO_PHASE      = PCT_NO_PHASE,
        PCT_MAX_PEAK      = PCT_MAX_PEAK,
        PCT_REF_PEAK      = PCT_REF_PEAK,
        PCT_PROM_PEAK     = PCT_PROM_PEAK,
        PCT_CENTER        = PCT_CENTER,
    )

    if not apply_phase:
        phi0_used = 0.0
        phi1_used = 0.0
        ref_used  = np.nan
        applied   = 0
    else:
        mix_complex, applied, phi0_used, phi1_used, ref_used = aplicar_fase(
            mix_complex,
            ppm_axis[:len(mix_complex)],
            ref_ppm    = float(ref_ppm_sel),
            prob       = 1.0,
            phi0_range = phi0_range,
            phi1_range = phi1_range,
        )

    mix_meta = {
        "mix_id"         : mix_id,
        "baseline_offset": baseline_offset,
        "noise_sigma"    : noise_sigma,
        "phase_applied"  : int(bool(applied)),
        "phi0"           : float(phi0_used),
        "phi1"           : float(phi1_used),
        "ref_ppm"        : float(ref_used) if np.isfinite(ref_used) else np.nan,
        "phase_mode"     : mode_tag,
    }

    return mix_complex, conc_dict, per_compound_rows, mix_meta


def generate_mixture_spectra(
    master_df          : pd.DataFrame,
    standardsSpec      : list,
    ppm_axis           : np.ndarray,
    v                  : np.ndarray,
    max_shift          : int,
    iterations         : int   = 2000,
    start_id           : int   = 0,
    standard_scale     : float = 1.0,
    aplicar_deformacion: bool  = False,
    gauss_sigma_range  : tuple = (0.0, 2.0),
    lorentz_gamma_range: tuple = (0.0, 2.0),
    asym_prob          : float = 0.2,
    asym_decay_pts     : int   = 30,
    asym_amp_range     : tuple = (0.0, 0.10),
    REF_REFERENCE_PPM  : float = None,
    PCT_NO_PHASE       : float = 0.30,
    PCT_MAX_PEAK       : float = 0.20,
    PCT_REF_PEAK       : float = 0.15,
    PCT_PROM_PEAK      : float = 0.20,
    PCT_CENTER         : float = 0.15,
    phi0_range         : tuple = (-15.0, 15.0),
    phi1_range         : tuple = (-3.0,   3.0),
    random_seed        : int   = 2,
) -> tuple:
    """
    Generates a dataset of synthetic NMR mixture spectra by calling  _generate_single_mixture() iteratively.  All distortion and phase parameters are forwarded to _generate_single_mixture().

    Parameters
            master_df           : pd.DataFrame                                     Merged metadata table from match_spectra_to_metadata().
        standardsSpec       : list of np.ndarray (float, shape=(N,2))          Aligned individual spectra from load_individual_spectra().
        ppm_axis            : np.ndarray (float, shape=(N,))                   Reference ppm axis.
        v                   : np.ndarray (float, shape=(N,))                   Internal standard signal from load_internal_standard().
        max_shift           : int                                              Maximum random shift in points per compound spectrum.
        iterations          : int                                              Number of synthetic mixtures to generate. Default: 2000.
        start_id            : int                                              Starting mix_id index. Default: 0.
        standard_scale      : float                                            Scaling factor for the internal standard. Default: 1.0.
        aplicar_deformacion : bool                                             If True, applies peak deformation to each compound. Default: False.
        gauss_sigma_range   : tuple of (float, float)                          Gaussian sigma range for deformar_picos(). Default: (0.0, 2.0).
        lorentz_gamma_range : tuple of (float, float)                          Lorentzian gamma range for deformar_picos(). Default: (0.0, 2.0).
        asym_prob           : float                                            Asymmetry probability for deformar_picos(). Default: 0.2.
        asym_decay_pts      : int                                              Asymmetry decay points for deformar_picos(). Default: 30.
        asym_amp_range      : tuple of (float, float)                          Asymmetry amplitude range for deformar_picos(). Default: (0.0, 0.10).
        REF_REFERENCE_PPM   : float or None                                    Fixed reference ppm for choose_phase_strategy(). Default: None.
        PCT_NO_PHASE        : float                                            Proportion with no phase correction. Default: 0.30.
        PCT_MAX_PEAK        : float                                            Proportion using max peak pivot. Default: 0.20.
        PCT_REF_PEAK        : float                                            Proportion using fixed reference ppm. Default: 0.15.
        PCT_PROM_PEAK       : float                                            Proportion using prominent random peak. Default: 0.20.
        PCT_CENTER          : float                                            Proportion using spectral center. Default: 0.15.
        phi0_range          : tuple of (float, float)                          Zero order phase range for aplicar_fase(). Default: (-15.0, 15.0).
        phi1_range          : tuple of (float, float)                          First-order phase range for aplicar_fase(). Default: (-3.0, 3.0).
        random_seed         : int                                               Seed for reproducibility. Default: 2.

    Returns
        spectra             : list of np.ndarray (complex, shape=(N,))        All generated complex spectra.
        concs               : list of dict                                    Concentration dicts per mixture.
        per_compound_params : list of dict                                    Per-compound logging rows (all mixtures concatenated).
        mix_metadata        : list of dict                                    Mixture-level metadata rows (baseline, noise, phase per mixture).

    Used by (export / online prediction pipeline)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    spectra             = []
    concs               = []
    per_compound_params = []
    mix_metadata        = []

    for k in range(iterations):
        mix_id = start_id + k

        mix_complex, conc_dict, per_compound_rows, mix_meta = _generate_single_mixture(
            mix_id              = mix_id,
            master_df           = master_df,
            standardsSpec       = standardsSpec,
            ppm_axis            = ppm_axis,
            v                   = v,
            max_shift           = max_shift,
            standard_scale      = standard_scale,
            aplicar_deformacion = aplicar_deformacion,
            gauss_sigma_range   = gauss_sigma_range,
            lorentz_gamma_range = lorentz_gamma_range,
            asym_prob           = asym_prob,
            asym_decay_pts      = asym_decay_pts,
            asym_amp_range      = asym_amp_range,
            REF_REFERENCE_PPM   = REF_REFERENCE_PPM,
            PCT_NO_PHASE        = PCT_NO_PHASE,
            PCT_MAX_PEAK        = PCT_MAX_PEAK,
            PCT_REF_PEAK        = PCT_REF_PEAK,
            PCT_PROM_PEAK       = PCT_PROM_PEAK,
            PCT_CENTER          = PCT_CENTER,
            phi0_range          = phi0_range,
            phi1_range          = phi1_range,
        )

        spectra.append(mix_complex)
        concs.append(conc_dict)
        per_compound_params.extend(per_compound_rows)
        mix_metadata.append(mix_meta)

    return spectra, concs, per_compound_params, mix_metadata

