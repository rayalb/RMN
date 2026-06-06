# =============================================================================
# preprocesador_dataset.py
# Pipeline for preprocessing synthetic NMR spectra and building
# (R, S, y) pair datasets for the pSCNN neural network.
#
# Depends on: generador_espectros.py
# =============================================================================

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

from generador_espectros import (
    load_individual_spectra,
    load_metabolic_profile,
    load_internal_standard,
    match_spectra_to_metadata,
    _generate_single_mixture,
)


# =============================================================================
# SECTION 1 — Baseline correction (airPLS, as in the paper)
# =============================================================================

def apply_binning(
    intensity  : np.ndarray,
    ppm        : np.ndarray,
    bin_factor : int = 1,
) -> tuple:
    """
    Reduces spectrum resolution by averaging every bin_factor consecutive points.
    Useful for reducing memory footprint while preserving peak shapes.

    Parameters
    ----------
    intensity  : np.ndarray (float, shape=(N,))
        Spectrum intensities.
    ppm        : np.ndarray (float, shape=(N,))
        Chemical shift axis in ppm.
    bin_factor : int
        Number of points to average. 1 = no binning. Default: 1.

    Returns
    -------
    intensity_binned : np.ndarray (float32, shape=(N//bin_factor,))
        Binned intensities.
    ppm_binned       : np.ndarray (float, shape=(N//bin_factor,))
        Binned ppm axis (mean of each bin).

    Used by
    -------
    preprocess_spectrum()
    """
    if bin_factor <= 1:
        return intensity, ppm

    n      = len(intensity)
    n_trim = (n // bin_factor) * bin_factor

    intensity_binned = (intensity[:n_trim]
                        .reshape(-1, bin_factor)
                        .mean(axis=1)
                        .astype("float32"))
    ppm_binned       = (ppm[:n_trim]
                        .reshape(-1, bin_factor)
                        .mean(axis=1))

    return intensity_binned, ppm_binned

def _whittaker_smooth(
    x          : np.ndarray,
    w          : np.ndarray,
    lambda_    : float,
    differences: int = 1
) -> np.ndarray:
    """
    Applies Whittaker smoother to signal x with weights w.
    Internal helper for airPLS baseline correction.

    Parameters
    ----------
    x           : np.ndarray (float, shape=(N,))
        Input signal.
    w           : np.ndarray (float, shape=(N,))
        Weight vector.
    lambda_     : float
        Smoothing parameter. Higher = smoother baseline.
    differences : int
        Order of differences. Default: 1.

    Returns
    -------
    np.ndarray (float, shape=(N,))
        Smoothed background estimate.

    Used by
    -------
    airPLS()
    """
    X = np.matrix(x)
    m = X.size
    E = eye(m, format="csc")
    for _ in range(differences):
        E = E[1:] - E[:-1]
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T)
    return np.array(spsolve(A, B))


def airPLS(
    x       : np.ndarray,
    lambda_ : float = 100,
    porder  : int   = 1,
    itermax : int   = 15
) -> np.ndarray:
    """
    Adaptive iteratively reweighted penalized least squares (airPLS)
    baseline correction, as used in the reference paper.

    Parameters
    ----------
    x       : np.ndarray (float, shape=(N,))
        Input spectrum signal.
    lambda_ : float
        Smoothing parameter. Default: 100.
    porder  : int
        Order of differences for smoothing. Default: 1.
    itermax : int
        Maximum number of iterations. Default: 15.

    Returns
    -------
    np.ndarray (float, shape=(N,))
        Estimated baseline of the input spectrum.

    Used by
    -------
    preprocess_spectrum()
    """
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z    = _whittaker_smooth(x, w, lambda_, porder)
        d    = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum()) or i == itermax:
            if i == itermax:
                warnings.warn(
                    "airPLS: maximum iterations reached — baseline may be inaccurate.",
                    UserWarning,
                    stacklevel=2
                )
            break
        w[d >= 0] = 0
        w[d < 0]  = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0]      = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1]     = w[0]
    return z


# =============================================================================
# SECTION 2 — Preprocessing helpers
# =============================================================================

def crop_ppm_range(
    ppm      : np.ndarray,
    intensity: np.ndarray,
    ppm_min  : float = 0.0,
    ppm_max  : float = 10.0
) -> tuple:
    """
    Crops the spectrum to a specified ppm window.

    Parameters
    ----------
    ppm       : np.ndarray (float, shape=(N,))
        Chemical shift axis in ppm.
    intensity : np.ndarray (float, shape=(N,))
        Spectrum intensities.
    ppm_min   : float
        Lower ppm boundary. Default: 0.0.
    ppm_max   : float
        Upper ppm boundary. Default: 10.0.

    Returns
    -------
    ppm_crop       : np.ndarray (float, shape=(M,))
        Cropped ppm axis.
    intensity_crop : np.ndarray (float, shape=(M,))
        Cropped intensities.

    Used by
    -------
    preprocess_spectrum()
    """
    mask = (ppm >= ppm_min) & (ppm <= ppm_max)
    return ppm[mask], intensity[mask]


def normalize_by_internal_standard(
    ppm              : np.ndarray,
    intensity        : np.ndarray,
    ppm_is_nominal   : float = 0.0,
    window           : float = 0.2,
    eps              : float = 1e-12
) -> np.ndarray:
    """
    Normalizes a spectrum by the height of the internal standard (IS) peak.
    Searches for the maximum absolute intensity within a window around
    the nominal IS ppm position.
    If the IS region is not found or the peak is below eps, returns
    the spectrum unchanged with a warning.

    Parameters
    ----------
    ppm            : np.ndarray (float, shape=(N,))
        Chemical shift axis in ppm.
    intensity      : np.ndarray (float, shape=(N,))
        Spectrum intensities.
    ppm_is_nominal : float
        Nominal ppm of the internal standard peak. Default: 0.0 (TSP).
    window         : float
        Half-width of the search window around ppm_is_nominal. Default: 0.2.
    eps            : float
        Minimum peak height to avoid division by zero. Default: 1e-12.

    Returns
    -------
    np.ndarray (float, shape=(N,))
        Normalized intensities. Unchanged if IS peak not found or too small.

    Used by
    -------
    preprocess_spectrum()
    """
    mask_is = (ppm >= ppm_is_nominal - window) & (ppm <= ppm_is_nominal + window)

    if not np.any(mask_is):
        warnings.warn(
            f"IS peak not found in window [{ppm_is_nominal - window:.2f}, "
            f"{ppm_is_nominal + window:.2f}] ppm. Returning unnormalized spectrum.",
            UserWarning,
            stacklevel=2
        )
        return intensity

    peak_height = np.max(np.abs(intensity[mask_is]))
    if peak_height < eps:
        warnings.warn(
            "IS peak height is below eps threshold. Returning unnormalized spectrum.",
            UserWarning,
            stacklevel=2
        )
        return intensity

    return intensity / peak_height


def preprocess_spectrum(
    ppm            : np.ndarray,
    intensity      : np.ndarray,
    ppm_is_nominal : float = 0.0,
    window_is      : float = 0.2,
    ppm_min        : float = 0.0,
    ppm_max        : float = 10.0,
    lambda_        : float = 100,
    porder         : int   = 1,
    itermax        : int   = 15,
    bin_factor     : int   = 1,
) -> tuple:
    """
    Full preprocessing pipeline for a single NMR spectrum:
        1. airPLS baseline correction.
        2. Baseline subtraction.
        3. Crop to [ppm_min, ppm_max] window.
        4. Normalization by internal standard peak.
        5. Binning (optional) — averages every bin_factor points.

    Parameters
    ----------
    ppm            : np.ndarray (float, shape=(N,))
        Chemical shift axis in ppm.
    intensity      : np.ndarray (float, shape=(N,))
        Raw spectrum intensities.
    ppm_is_nominal : float
        Nominal ppm of the internal standard. Default: 0.0 (TSP).
    window_is      : float
        Half-width of IS search window. Default: 0.2.
    ppm_min        : float
        Lower ppm crop boundary. Default: 0.0.
    ppm_max        : float
        Upper ppm crop boundary. Default: 10.0.
    lambda_        : float
        airPLS smoothing parameter. Default: 100.
    porder         : int
        airPLS difference order. Default: 1.
    itermax        : int
        airPLS maximum iterations. Default: 15.
    bin_factor     : int
        Binning factor — average every bin_factor points. Default: 1 (no binning).
        Recommended values: 1 (none), 2, 4. Max safe value depends on peak width.

    Returns
    -------
    ppm_out : np.ndarray (float, shape=(M,))
        Preprocessed ppm axis (cropped, optionally binned).
    int_out : np.ndarray (float32, shape=(M,))
        Preprocessed intensities (baseline-corrected, cropped, normalized,
        optionally binned).

    Used by
    -------
    load_individual_preprocessed(), preprocess_single_mixture()
    """
    # 1) Baseline correction
    baseline       = airPLS(intensity.astype(float), lambda_=lambda_,
                            porder=porder, itermax=itermax)
    intensity_corr = intensity - baseline

    # 2) Crop
    ppm_crop, int_crop = crop_ppm_range(ppm, intensity_corr, ppm_min, ppm_max)

    # 3) Normalize by IS
    int_norm = normalize_by_internal_standard(
        ppm_crop, int_crop,
        ppm_is_nominal=ppm_is_nominal,
        window=window_is
    )

    # 4) Binning
    int_norm, ppm_crop = apply_binning(int_norm, ppm_crop, bin_factor)

    return ppm_crop, int_norm.astype("float32")


# =============================================================================
# SECTION 3 — Load and preprocess individual spectra from disk
# =============================================================================

def load_individual_preprocessed(
    compound_name  : str,
    ind_dir        : str,
    comp_to_file   : dict,
    ppm_is_nominal : float = 0.0,
    window_is      : float = 0.2,
    ppm_min        : float = 0.0,
    ppm_max        : float = 10.0,
    bin_factor     : int   = 1,
) -> tuple:
    """
    Loads a single individual compound spectrum from disk and applies
    the full preprocessing pipeline including optional binning.

    Parameters
    ----------
    compound_name  : str
    ind_dir        : str
    comp_to_file   : dict
    ppm_is_nominal : float — Default: 0.0.
    window_is      : float — Default: 0.2.
    ppm_min        : float — Default: 0.0.
    ppm_max        : float — Default: 10.0.
    bin_factor     : int   — Binning factor. Default: 1 (no binning).

    Returns
    -------
    ppm_prep : np.ndarray (float, shape=(M,))
    int_prep : np.ndarray (float32, shape=(M,))

    Used by
    -------
    build_individual_cache()
    """
    filename = comp_to_file[compound_name]
    path     = os.path.join(ind_dir, filename)

    df        = pd.read_csv(path, sep="\t")
    ppm       = df.iloc[:, 0].values.astype(float)
    intensity = df.iloc[:, 1].values.astype(float)

    ppm_prep, int_prep = preprocess_spectrum(
        ppm, intensity,
        ppm_is_nominal = ppm_is_nominal,
        window_is      = window_is,
        ppm_min        = ppm_min,
        ppm_max        = ppm_max,
        bin_factor     = bin_factor,
    )
    return ppm_prep, int_prep


def load_metadata_individual(meta_ind_path: str) -> tuple:
    """
    Loads the individual spectra metadata Excel and builds the
    compound-to-filename mapping.

    Parameters
    ----------
    meta_ind_path : str
        Full path to the individual spectra metadata Excel file.
        Expected columns: 'nombre_compuesto', 'nombre_archivo_csv'.

    Returns
    -------
    meta_ind     : pd.DataFrame
        Full metadata table.
    comp_to_file : dict
        Mapping {nombre_compuesto (str): nombre_archivo_csv (str)}.

    Used by
    -------
    build_individual_cache()
    """
    meta_ind     = pd.read_excel(meta_ind_path)
    comp_to_file = {
        row["nombre_compuesto"]: row["nombre_archivo_csv"]
        for _, row in meta_ind.iterrows()
    }
    return meta_ind, comp_to_file


def build_individual_cache(
    meta_ind       : pd.DataFrame,
    ind_dir        : str,
    comp_to_file   : dict,
    ppm_is_nominal : float = 0.0,
    window_is      : float = 0.2,
    ppm_min        : float = 0.0,
    ppm_max        : float = 10.0,
    bin_factor     : int   = 1,
) -> dict:
    """
    Preloads and preprocesses all individual spectra into a cache dict.
    Avoids reloading from disk on every pair generation.

    Parameters
    ----------
    meta_ind       : pd.DataFrame
    ind_dir        : str
    comp_to_file   : dict
    ppm_is_nominal : float — Default: 0.0.
    window_is      : float — Default: 0.2.
    ppm_min        : float — Default: 0.0.
    ppm_max        : float — Default: 10.0.
    bin_factor     : int   — Binning factor. Default: 1 (no binning).
                             Must match bin_factor used for mixture preprocessing.

    Returns
    -------
    ind_cache : dict
        {compound_name (str): int_prep (np.ndarray float32, shape=(M,))}

    Used by
    -------
    build_dataset_split()
    """
    ind_cache = {}
    for compound_name in meta_ind["nombre_compuesto"]:
        try:
            _, int_prep = load_individual_preprocessed(
                compound_name  = compound_name,
                ind_dir        = ind_dir,
                comp_to_file   = comp_to_file,
                ppm_is_nominal = ppm_is_nominal,
                window_is      = window_is,
                ppm_min        = ppm_min,
                ppm_max        = ppm_max,
                bin_factor     = bin_factor,
            )
            ind_cache[compound_name] = int_prep
        except Exception as e:
            warnings.warn(
                f"Could not load individual spectrum for '{compound_name}': {e}",
                UserWarning,
                stacklevel=2
            )
    return ind_cache


# =============================================================================
# SECTION 4 — Preprocess a synthetic mixture from _generate_single_mixture
# =============================================================================

def preprocess_single_mixture(
    mix_complex    : np.ndarray,
    ppm_axis       : np.ndarray,
    ppm_is_nominal : float = 0.0,
    window_is      : float = 0.2,
    ppm_min        : float = 0.0,
    ppm_max        : float = 10.0,
    lambda_        : float = 100,
    porder         : int   = 1,
    itermax        : int   = 15,
    bin_factor     : int   = 1,
) -> tuple:
    """
    Extracts the real part of a complex synthetic mixture spectrum
    and applies the full preprocessing pipeline including optional binning.

    Parameters
    ----------
    mix_complex    : np.ndarray (complex, shape=(N,))
    ppm_axis       : np.ndarray (float, shape=(N,))
    ppm_is_nominal : float — Default: 0.0.
    window_is      : float — Default: 0.2.
    ppm_min        : float — Default: 0.0.
    ppm_max        : float — Default: 10.0.
    lambda_        : float — Default: 100.
    porder         : int   — Default: 1.
    itermax        : int   — Default: 15.
    bin_factor     : int   — Binning factor. Default: 1 (no binning).
                             Must match bin_factor used for individual spectra.

    Returns
    -------
    ppm_out : np.ndarray (float, shape=(M,))
    int_out : np.ndarray (float32, shape=(M,))

    Used by
    -------
    build_dataset_split()
    """
    real_part = mix_complex.real.astype(float)
    n         = min(len(real_part), len(ppm_axis))

    return preprocess_spectrum(
        ppm_axis[:n], real_part[:n],
        ppm_is_nominal = ppm_is_nominal,
        window_is      = window_is,
        ppm_min        = ppm_min,
        ppm_max        = ppm_max,
        lambda_        = lambda_,
        porder         = porder,
        itermax        = itermax,
        bin_factor     = bin_factor,
    )


# =============================================================================
# SECTION 5 — Build (R, S, y) pair dataset
# =============================================================================

def build_dataset_split(
    n_mixtures        : int,
    master_df         : pd.DataFrame,
    standardsSpec     : list,
    ppm_axis          : np.ndarray,
    v                 : np.ndarray,
    ind_cache         : dict,
    ind_to_master_idx : dict,
    max_shift         : int   = 60,
    ppm_is_nominal    : float = 0.0,
    window_is         : float = 0.2,
    ppm_min           : float = 0.0,
    ppm_max           : float = 10.0,
    bin_factor        : int   = 1,
    start_id          : int   = 0,
    mixture_kwargs    : dict  = None,
) -> dict:
    """
    Generates n_mixtures synthetic spectra on-the-fly and builds
    the (R, S, y) pair dataset for one split (train, valid, or test).

    For each mixture:
        - Generates a synthetic spectrum via _generate_single_mixture().
        - Preprocesses the real part via preprocess_single_mixture().
        - For each compound in ind_cache:
            * R = preprocessed individual spectrum.
            * S = preprocessed mixture spectrum.
            * y = 1 if compound present in mixture, 0 if absent.

    Only compounds present in both ind_cache AND ind_to_master_idx
    are used to build pairs. The remaining compounds in master_df
    still participate in mixture generation but are not paired.

    Parameters
    ----------
    n_mixtures        : int
        Number of synthetic mixtures to generate for this split.
    master_df         : pd.DataFrame
        Output of match_spectra_to_metadata(). Must have columns:
        'idx', 'Compuesto', 'p_ausencia', 'Media', 'Desvío',
        'Distribución recomendada'.
    standardsSpec     : list of np.ndarray (float, shape=(N,2))
        Aligned individual spectra from load_individual_spectra().
    ppm_axis          : np.ndarray (float, shape=(N,))
        Reference ppm axis.
    v                 : np.ndarray (float, shape=(N,))
        Internal standard signal from load_internal_standard().
    ind_cache         : dict
        {compound_name (str): preprocessed spectrum (np.ndarray float32)}
        from build_individual_cache(). Contains only the compounds
        used for pairing (subset of master_df).
    ind_to_master_idx : dict
        {compound_name (str): idx (int)} mapping from ind_cache compound
        names to their corresponding idx in standardsSpec/conc_dict.
        Built via normalize_name() matching. Required because compound
        names in ind_cache and master_df may differ in case/accents.
    max_shift         : int
        Maximum shift in points for _generate_single_mixture(). Default: 60.
    ppm_is_nominal    : float
        Nominal ppm of the IS for preprocessing. Default: 0.0.
    window_is         : float
        Half-width of IS search window. Default: 0.2.
    ppm_min           : float
        Lower ppm crop boundary. Default: 0.0.
    ppm_max           : float
        Upper ppm crop boundary. Default: 10.0.
    start_id          : int
        Starting mix_id for _generate_single_mixture(). Default: 0.
    **kwargs
        Any additional parameter accepted by _generate_single_mixture()
        (baseline_range, noise_level, aplicar_deformacion,
        phi0_range, phi1_range, PCT_NO_PHASE, etc.)

    Returns
    -------
    dict with keys:
        "R"      : np.ndarray (float32, shape=(N_pairs, M))
            Individual spectra (one per paired compound per mixture).
        "S"      : np.ndarray (float32, shape=(N_pairs, M))
            Mixture spectra (repeated for each compound pair).
        "y"      : np.ndarray (float32, shape=(N_pairs, 1))
            Binary labels: 1 = compound present, 0 = absent.
        "comp"   : np.ndarray (str,     shape=(N_pairs,))
            Compound name for each pair.
        "mix_id" : np.ndarray (int,     shape=(N_pairs,))
            Local mixture index (0..n_mixtures-1) for each pair.

    Used by
    -------
    build_full_dataset()
    """
    if mixture_kwargs is None:
        mixture_kwargs = {}

    R_list           = []
    S_list           = []
    y_list           = []
    comp_list        = []
    mix_id_list      = []
    mix_composition  = {}   # {mix_local_id: {comp_name: conc, ...}}

    # Build reverse mapping: idx → comp_name (for all compounds in master_df)
    idx_to_compname = {}
    for _, row in master_df.iterrows():
        idx_to_compname[int(row["idx"])] = row["Compuesto"]

    for mix_local_id in tqdm(range(n_mixtures), desc="Generating pairs (R, S, y)"):
        # 1) Generate synthetic mixture
        mix_complex, conc_dict, _, _ = _generate_single_mixture(
            mix_id        = start_id + mix_local_id,
            master_df     = master_df,
            standardsSpec = standardsSpec,
            ppm_axis      = ppm_axis,
            v             = v,
            max_shift     = max_shift,
            **mixture_kwargs                          # only mixture params here
        )

        # Store full composition of this mixture (all compounds, not just paired ones)
        mix_composition[mix_local_id] = {
            idx_to_compname[idx]: float(conc)
            for idx, conc in conc_dict.items()
            if idx != "_IS" and idx in idx_to_compname
        }

        # 2) Preprocess mixture real part
        _, mix_prep = preprocess_single_mixture(
            mix_complex    = mix_complex,
            ppm_axis       = ppm_axis,
            ppm_is_nominal = ppm_is_nominal,
            window_is      = window_is,
            ppm_min        = ppm_min,
            ppm_max        = ppm_max,
            bin_factor     = bin_factor,
        )

        # 3) Build one pair per compound in ind_cache
        for comp_name, ind_spec in ind_cache.items():
            if comp_name not in ind_to_master_idx:
                continue   # skip if no idx mapping found

            idx   = ind_to_master_idx[comp_name]
            y_val = 1.0 if idx in conc_dict else 0.0

            R_list.append(ind_spec)
            S_list.append(mix_prep)
            y_list.append(y_val)
            comp_list.append(comp_name)
            mix_id_list.append(mix_local_id)

    R = np.stack(R_list, axis=0).astype("float32")
    S = np.stack(S_list, axis=0).astype("float32")
    y = np.array(y_list, dtype="float32").reshape(-1, 1)

    return {
        "R"              : R,
        "S"              : S,
        "y"              : y,
        "comp"           : np.array(comp_list),
        "mix_id"         : np.array(mix_id_list),
        "mix_composition": mix_composition,   # {mix_id: {comp_name: conc}}
    }


# =============================================================================
# SECTION 6 — Build full train/valid/test dataset
# =============================================================================

def build_full_dataset(
    master_df         : pd.DataFrame,
    standardsSpec     : list,
    ppm_axis          : np.ndarray,
    v                 : np.ndarray,
    ind_cache         : dict,
    ind_to_master_idx : dict,
    n_total           : int   = 2000,
    train_frac        : float = 0.70,
    valid_frac        : float = 0.15,
    max_shift         : int   = 60,
    ppm_is_nominal    : float = 0.0,
    window_is         : float = 0.2,
    ppm_min           : float = 0.0,
    ppm_max           : float = 10.0,
    bin_factor        : int   = 1,
    random_seed       : int   = 42,
    mixture_kwargs    : dict  = None,
) -> tuple:
    """
    Builds train, validation, and test (R, S, y) pair datasets
    by generating synthetic mixtures on-the-fly for each split.

    Split sizes are computed from n_total and train_frac/valid_frac.
    test_frac = 1 - train_frac - valid_frac.

    Parameters
    ----------
    master_df         : pd.DataFrame
        Output of match_spectra_to_metadata().
    standardsSpec     : list of np.ndarray (float, shape=(N,2))
        Aligned individual spectra from load_individual_spectra().
    ppm_axis          : np.ndarray (float, shape=(N,))
        Reference ppm axis.
    v                 : np.ndarray (float, shape=(N,))
        Internal standard signal from load_internal_standard().
    ind_cache         : dict
        Preprocessed individual spectra from build_individual_cache().
    ind_to_master_idx : dict
        {compound_name (str): idx (int)} mapping from ind_cache names
        to their idx in standardsSpec/conc_dict.
        Built via normalize_name() matching in the calling notebook.
    n_total           : int
        Total number of synthetic mixtures to generate. Default: 2000.
    train_frac        : float
        Fraction for training set. Default: 0.70.
    valid_frac        : float
        Fraction for validation set. Default: 0.15.
    max_shift         : int
        Maximum shift in points. Default: 60.
    ppm_is_nominal    : float
        Nominal ppm of the IS. Default: 0.0.
    window_is         : float
        Half-width of IS search window. Default: 0.2.
    ppm_min           : float
        Lower ppm crop boundary. Default: 0.0.
    ppm_max           : float
        Upper ppm crop boundary. Default: 10.0.
    random_seed       : int
        Seed for reproducibility. Default: 42.
    **kwargs
        Any additional parameter for _generate_single_mixture().

    Returns
    -------
    aug_train : dict — train split  {R, S, y, comp, mix_id}
    aug_valid : dict — valid split  {R, S, y, comp, mix_id}
    aug_test  : dict — test split   {R, S, y, comp, mix_id}

    Used by
    -------
    (neural network training pipeline)
    """
    np.random.seed(random_seed)

    if mixture_kwargs is None:
        mixture_kwargs = {}

    n_train = int(n_total * train_frac)
    n_valid = int(n_total * valid_frac)
    n_test  = n_total - n_train - n_valid

    print(f"Split sizes — train: {n_train} | valid: {n_valid} | test: {n_test}")

    shared = dict(
        master_df         = master_df,
        standardsSpec     = standardsSpec,
        ppm_axis          = ppm_axis,
        v                 = v,
        ind_cache         = ind_cache,
        ind_to_master_idx = ind_to_master_idx,
        max_shift         = max_shift,
        ppm_is_nominal    = ppm_is_nominal,
        window_is         = window_is,
        ppm_min           = ppm_min,
        ppm_max           = ppm_max,
        bin_factor        = bin_factor,
        mixture_kwargs    = mixture_kwargs,
    )

    print("\n── Building TRAIN split ──")
    aug_train = build_dataset_split(n_mixtures=n_train, start_id=0, **shared)

    print("\n── Building VALID split ──")
    aug_valid = build_dataset_split(n_mixtures=n_valid, start_id=n_train, **shared)

    print("\n── Building TEST split ──")
    aug_test  = build_dataset_split(n_mixtures=n_test,  start_id=n_train + n_valid, **shared)

    print(f"\nTrain — R:{aug_train['R'].shape}  S:{aug_train['S'].shape}  y:{aug_train['y'].shape}")
    print(f"Valid — R:{aug_valid['R'].shape}  S:{aug_valid['S'].shape}  y:{aug_valid['y'].shape}")
    print(f"Test  — R:{aug_test['R'].shape}   S:{aug_test['S'].shape}   y:{aug_test['y'].shape}")

    return aug_train, aug_valid, aug_test


# =============================================================================
# ENTRY POINT — edit parameters here and run
# =============================================================================

if __name__ == "__main__":
    import pickle

    # ── General ──────────────────────────────────────────────────────────────
    ROOT_DIR    = "/home/ray/Documents/Quimica/"   # <-- change this
    OUTPUT_DIR  = os.path.join(ROOT_DIR, "aug")
    RANDOM_SEED = 42

    # ── File paths (individual spectra metadata) ──────────────────────────────
    META_IND_PATH = os.path.join(ROOT_DIR, "DataSet",
                                 "metadata_espectros_individuales_con_TSP sodico(SI).xlsx")
    IND_DIR       = os.path.join(ROOT_DIR, "Espectros individuales + TSP sodico (SI)")

    # ── File paths (generador_espectros inputs) ───────────────────────────────
    EXCEL_FILENAME      = "Perfil metabolico vino.xlsx"
    EXCEL_SHEET         = "General"
    INTERNAL_STD_FILE   = "tsp-d4 sodico.csv"
    INTERNAL_STD_FOLDER = "Moleculas mol"
    SPECTRA_SUBFOLDER   = os.path.join("Moleculas mol", "Espectros individuales")
    SPECTRA_EXT         = ".csv"

    # ── Dataset split ─────────────────────────────────────────────────────────
    N_TOTAL    = 5
    TRAIN_FRAC = 0.70
    VALID_FRAC = 0.15   # test_frac = 0.15

    # ── Preprocessing ─────────────────────────────────────────────────────────
    PPM_IS    = 0.0     # nominal ppm of internal standard (TSP)
    WINDOW_IS = 0.2     # +/- window around IS peak
    PPM_MIN   = 0.0     # crop range
    PPM_MAX   = 10.0

    # ── Mixture generation (passed to _generate_single_mixture) ──────────────
    MAX_SHIFT_PTS       = 60
    APLICAR_DEFORMACION = False
    BASELINE_RANGE      = (-0.02, 0.02)
    NOISE_LEVEL         = 0.02
    REF_REFERENCE_PPM   = None
    PCT_NO_PHASE        = 0.30
    PCT_MAX_PEAK        = 0.20
    PCT_REF_PEAK        = 0.15
    PCT_PROM_PEAK       = 0.20
    PCT_CENTER          = 0.15
    PHI0_RANGE          = (-15.0, 15.0)
    PHI1_RANGE          = (-3.0,   3.0)

    # ── Pipeline ─────────────────────────────────────────────────────────────

    # 1) Load generador_espectros data
    standardsSpec, standardsDictionary, x_ref = load_individual_spectra(
        root_dir  = ROOT_DIR,
        subfolder = SPECTRA_SUBFOLDER,
        ext       = SPECTRA_EXT,
    )
    df = load_metabolic_profile(root_dir=ROOT_DIR, filename=EXCEL_FILENAME,
                                sheet_name=EXCEL_SHEET)
    ppm_std, v_std = load_internal_standard(root_dir=ROOT_DIR,
                                            filename=INTERNAL_STD_FILE,
                                            subfolder=INTERNAL_STD_FOLDER)
    master_df, diagnostics = match_spectra_to_metadata(
        df=df, standardsDictionary=standardsDictionary)
    ppm_axis = x_ref

    print(f"Matched: {diagnostics['matched']} / {diagnostics['total']}")

    # 2) Load individual spectra metadata and build cache
    meta_ind, comp_to_file = load_metadata_individual(META_IND_PATH)
    ind_cache = build_individual_cache(
        meta_ind       = meta_ind,
        ind_dir        = IND_DIR,
        comp_to_file   = comp_to_file,
        ppm_is_nominal = PPM_IS,
        window_is      = WINDOW_IS,
        ppm_min        = PPM_MIN,
        ppm_max        = PPM_MAX,
    )
    print(f"Individual spectra cached: {len(ind_cache)}")

    # 3) Build full dataset
    aug_train, aug_valid, aug_test = build_full_dataset(
        master_df           = master_df,
        standardsSpec       = standardsSpec,
        ppm_axis            = ppm_axis,
        v                   = v_std,
        ind_cache           = ind_cache,
        n_total             = N_TOTAL,
        train_frac          = TRAIN_FRAC,
        valid_frac          = VALID_FRAC,
        max_shift           = MAX_SHIFT_PTS,
        ppm_is_nominal      = PPM_IS,
        window_is           = WINDOW_IS,
        ppm_min             = PPM_MIN,
        ppm_max             = PPM_MAX,
        random_seed         = RANDOM_SEED,
        aplicar_deformacion = APLICAR_DEFORMACION,
        baseline_range      = BASELINE_RANGE,
        noise_level         = NOISE_LEVEL,
        REF_REFERENCE_PPM   = REF_REFERENCE_PPM,
        PCT_NO_PHASE        = PCT_NO_PHASE,
        PCT_MAX_PEAK        = PCT_MAX_PEAK,
        PCT_REF_PEAK        = PCT_REF_PEAK,
        PCT_PROM_PEAK       = PCT_PROM_PEAK,
        PCT_CENTER          = PCT_CENTER,
        phi0_range          = PHI0_RANGE,
        phi1_range          = PHI1_RANGE,
    )
    '''
    # 4) Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, data in [("train", aug_train), ("valid", aug_valid), ("test", aug_test)]:
        path = os.path.join(OUTPUT_DIR, f"data_augment_{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved: {path}")
    '''