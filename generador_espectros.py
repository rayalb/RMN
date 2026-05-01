import numpy as np
import pandas as pd
import nmrglue as nmr 

import os 



# Helpers for Load individual spectras

def _to_float_series(s: pd.Series) -> pd.Series:
    """
    Converts a pandas Series to numeric values, fixing comma decimals and coercing invalid entries to NaN. 
    Used for def load_and_align_spectra
    """

    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def read_spectrum_file(path: str):
    """
    Reads a spectrum file in flexible formats and extracts ppm, real, and imaginary signals as clean numeric arrays.
    If the input spectrum does not contain an imaginary part, it automatically creates an imag array filled with zeros and sets has_imag = False.'
    Used for def load_and_align_spectra
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
        parts = s.str.split(",", n=2, expand=True) if s.str.contains(",", regex=False).any() else s.str.split(r"\s+", n=2, expand=True)
        if parts.shape[1] < 2:
            raise ValueError(f"{os.path.basename(path)}: only one column. There is no spectrum in the file.")

        ppm  = _to_float_series(parts.iloc[:, 0])
        real = _to_float_series(parts.iloc[:, 1])

        if parts.shape[1] >= 3:
            imag = _to_float_series(parts.iloc[:, 2]).fillna(0.0)
            has_imag = True
        else:
            imag = pd.Series(np.zeros(len(ppm)))
            has_imag = False

        df2 = pd.DataFrame({"ppm": ppm, "real": real, "imag": imag}).dropna(subset=["ppm","real"])
        return df2["ppm"].to_numpy(float), df2["real"].to_numpy(float), df2["imag"].to_numpy(float), has_imag

    raise ValueError(f"{os.path.basename(path)}: an expected number of columns ({df.shape[1]}).")

def resample_to_ref(ppm_i_desc, y_i_desc, ppm_ref_desc):
    """
    Interpolates a spectrum onto a reference ppm axis, handling descending RMN order (10 → 0)
    by internally reversing the arrays before interpolation and restoring the original order afterward.
    Used for def load_and_align_spectra
    """
    x_i_inc = ppm_i_desc[::-1]
    y_i_inc = y_i_desc[::-1]
    x_ref_inc = ppm_ref_desc[::-1]

    f = interp1d(
        x_i_inc, y_i_inc,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True
    )
    y_ref_inc = f(x_ref_inc)
    return y_ref_inc[::-1]  

def load_individual_spectra( root_dir: str, subfolder: str = os.path.join("Moleculas mol", "Espectros individuales"), 
                            ext: str = ".csv", atol: float = 1e-6):
    """
    Loads spectrum files, extracts ppm/real signals, and aligns all spectra to a common ppm axis.
    Parameters
    ----------
    root_dir : [str] Base directory path.
    subfolder : [str] Relative path to the spectra folder.
    ext : [str] File extension to filter (default: ".csv").
    atol : [float]  Absolute tolerance for ppm axis comparison.

    Returns
    -------
    standardsSpec : [list of np.ndarray (float, shape=(N,2))] List of aligned spectra [ppm, real].
    standardsDictionary : [list of dict] Metadata per spectrum.
    x_ref : [np.ndarray (float)] Reference ppm axis used for alignment.
    """

    folder = os.path.join(root_dir, path_to_spectras_ind)

    filenames = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(ext) and not f.startswith("~$")
    ])

    standardsSpec = []
    standardsDictionary = []

    # --- Load csv file of NMR spectra---
    for file in filenames:
        ppm, real, imag, has_imag = read_spectrum_file(os.path.join(folder, file))

        standardsSpec.append(np.column_stack((ppm, real)))

        standardsDictionary.append({"filename": file,"points_original": len(ppm),"has_imag": has_imag,"resampled": False})

    if len(standardsSpec) == 0:
        raise ValueError("No spectra files found.")

    # --- Define reference axis for NMR spectras---
    idx_ref = int(np.argmax([spec.shape[0] for spec in standardsSpec]))
    x_ref = standardsSpec[idx_ref][:, 0]

    # --- Resample ---
    for i in range(len(standardsSpec)):
        x_i = standardsSpec[i][:, 0]
        y_i = standardsSpec[i][:, 1]

        need_resample = (
            len(x_i) != len(x_ref)
            or not np.allclose(x_i, x_ref, rtol=0, atol=atol)
        )

        if need_resample:
            y_res = resample_to_ref(x_i, y_i, x_ref)
            standardsSpec[i] = np.column_stack((x_ref, y_res))
            standardsDictionary[i]["resampled"] = True

    return standardsSpec, standardsDictionary, x_ref


def chequear():
    ##### ....
    return None 

def cargar_perfil_metobolico(path = '/home/ray/Documents/...', show = False):
    '''
    Comentarios
    '''
    df = pd.read_excel(path, sheet_name="General")

    # Chequeo
    nombres_bien = chequear()

    if show == True:
        print('Ok!')

    return df 

def definir_standard_interno(path, compuesto = 'metanol'):
 # Lectura del estándar interno 

path_std = os.path.join(root_dir, "Moleculas mol", "tsp-d4 sodico.csv")

# Lectura robusta del estándar interno (acepta tab, coma, 1–3 columnas)
df_std = pd.read_csv(path_std, sep="\t", header=None, engine="python")

# Si vino todo en una sola columna (caso Excel: "ppm,real")
if df_std.shape[1] == 1:
    parts = df_std.iloc[:, 0].astype(str)

    # separar por coma o espacios
    if parts.str.contains(",", regex=False).any():
        parts = parts.str.split(",", expand=True)
    else:
        parts = parts.str.split(r"\s+", expand=True)

    df_std = parts

# Ahora esperamos al menos 2 columnas: ppm | real | (imag opcional)
ppm_std  = pd.to_numeric(df_std.iloc[:, 0], errors="coerce").to_numpy()
v        = pd.to_numeric(df_std.iloc[:, 1], errors="coerce").to_numpy()

# Limpieza básica
mask = np.isfinite(ppm_std) & np.isfinite(v)
ppm_std = ppm_std[mask]
v       = v[mask]

return None



def entra_en_mezcla(p_ausencia):
    return np.random.rand() > float(p_ausencia)
#Función para decidir la inclusión de un compuesto en la mezcla según su probabilidad de ausencia desde el perfil metabolico


def sample_conc(media, desvio, distribucion):
# Función para muestrear la concentración de un compuesto según distribución estadística desde el perfil metabolico. 
#La presente funcion contempla compuestos cuya distribucion estadistica de concentracion sigan distribuciones:  
#normales, lognormales, gamma, rectangulares y triangulares

    m, s = float(media), float(desvio)

    # normaliza el texto a minúsculas y quita guiones / espacios
    dist = str(distribucion).strip().lower()
    dist_key = re.sub(r"[\s\-]", "", dist)
   
    if dist.startswith("normal"):
        return max(0.0, np.random.normal(m, s))
    
    if dist_key.startswith("lognormal"):
        if m <= 0 or s <= 0:
            return max(0.0, m)
        var = s**2
        sigma2 = np.log(1.0 + var/(m**2))
        sigma  = np.sqrt(max(1e-12, sigma2))
        mu     = np.log(m) - 0.5*sigma2
        x = np.random.lognormal(mean=mu, sigma=sigma)
        return max(0.0, float(x))

    if dist_key.startswith("gamma"):
        if m <= 0 or s <= 0:
            return max(0.0, m)
        k = (m/s)**2          # shape
        theta = (s**2)/m      # scale
        x = np.random.gamma(shape=max(1e-12, k), scale=max(1e-12, theta))
        return max(0.0, float(x))

    if dist.startswith("rectangular"):
        w = s * np.sqrt(12.0)
        a, b = max(0.0, m - w/2), max(m + w/2, m - w/2 + 1e-12)
        return np.random.uniform(a, b)

    if dist.startswith("triangular"):
        h = s * np.sqrt(6.0)
        a, c = max(0.0, m - h), max(m + h, m - h + 1e-12)
        return np.random.triangular(a, m, c)  

    return max(0.0, m)


def corrimiento_aleatorio(spec, max_shift):
# Función para aplicar un corrimiento aleatorio en puntos al espectro
    shift_pts = int(np.random.normal(0, max_shift/3))
    shift_pts = np.clip(shift_pts, -max_shift, max_shift)
    spec_shifted = ng.process.proc_base.cs(spec, shift_pts)
    return spec_shifted, shift_pts

def aplicar_fase(y_complex, x_ppm, ref_ppm=0.0, prob=1,phi0_range=(-15, 15), phi1_range=(-3, 3)):    # grados/ppm
    
    # --- sanity checks ---
    if not (0.0 <= float(prob) <= 1.0): raise ValueError("prob debe estar entre 0 y 1.")
    if not np.iscomplexobj(y_complex): raise ValueError("y_complex debe ser COMPLEJO (usa hilbert antes).")

    y_complex = np.asarray(y_complex)
    x_ppm     = np.asarray(x_ppm, dtype=float)

    if y_complex.shape != x_ppm.shape:raise ValueError("y_complex y x_ppm deben tener la MISMA forma.")

    aplicar = (np.random.rand() < prob)

    # Parámetros de fase
    phi0 = np.random.uniform(*phi0_range) if aplicar else 0.0
    phi1 = np.random.uniform(*phi1_range) if aplicar else 0.0
    phi_rad = np.deg2rad(phi0 + phi1 * (x_ppm - ref_ppm))  # shape (N,)
    y_out = y_complex * np.exp(1j * phi_rad)               # rotación compleja

    return y_out, int(aplicar), float(phi0), float(phi1), float(ref_ppm)

def deformar_picos(spec,gauss_sigma_range=(0.0, 2.0),lorentz_gamma_range=(0.0, 2.0),asym_prob=0.2,asym_decay_pts=30,asym_amp_range=(0.0, 0.10)):   
       
    y = spec

# ensanchamiento gaussiano en puntos
# ensanchamiento lorentziano en puntos
# probabilidad de asimetría
# decaimiento de la cola (puntos)
# amplitud relativa de la cola


    #  Gaussiano
    sigma = np.random.uniform(*gauss_sigma_range)
    if sigma > 0:
        L = max(3, int(6 * sigma) + 1)
        x = np.arange(-(L // 2), L // 2 + 1)
        g = np.exp(-0.5 * (x / sigma) ** 2)
        g /= g.sum()
        y = np.convolve(y, g, mode='same')

    #  Lorentziano
    gamma = np.random.uniform(*lorentz_gamma_range)
    if gamma > 0:
        L = max(3, int(10 * gamma) + 1)
        x = np.arange(-(L // 2), L // 2 + 1)
        l = 1.0 / (1.0 + (x / gamma) ** 2)
        l /= l.sum()
        y = np.convolve(y, l, mode='same')

    #  Asimetría
    asym_applied = 0   # 0 = no aplicada, 1 = aplicada
    amp = 0.0
    dir_sign = 0       # 1 derecha, -1 izquierda, 0 = sin asimetría

    if np.random.rand() < asym_prob and asym_decay_pts > 0:
        asym_applied = 1
        amp = np.random.uniform(*asym_amp_range)
        dir_sign = np.random.choice([-1, 1])

        L = max(3, int(5 * asym_decay_pts))
        x = np.arange(0, L)
        e = np.exp(-x / float(asym_decay_pts))
        e /= e.sum()

        # núcleo: delta + pequeña cola a un lado
        k = np.zeros(2 * L + 1)
        k[L] = 1.0 - amp
        if dir_sign > 0:
            k[L+1:L+1+L] = amp * e               # cola a la derecha
        else:
            k[L-L:L] = (amp * e)[::-1]           # cola a la izquierda

        y = np.convolve(y, k, mode='same')

    # 📌 parámetros de deformación para loguear en el Excel
    deform_params = {"gauss_sigma": float(sigma),"lorentz_gamma": float(gamma),"asym_applied": int(asym_applied),"asym_amp": float(amp),"asym_dir": int(dir_sign),"asym_decay_pts": int(asym_decay_pts)}

    return y, deform_params

def _ppm_center(ppm_axis):
    return float((np.max(ppm_axis) + np.min(ppm_axis)) / 2.0)
# Funcion auxiliar de deformacion de fase, para cambio de fase desde el centro del espectro

def _ref_from_max_peak(ppm_axis, y):
    """Pivote = ppm del pico de mayor intensidad absoluta."""
    idx = int(np.argmax(np.abs(y)))
    return float(ppm_axis[idx])
# Funcion auxiliar de deformacion de fase, para cambio de fase desde el pico de mayor intensidad de todos los espectro

def _ref_from_prominent_random(ppm_axis, y, q=0.90):
        yabs = np.abs(y)
    thr = np.quantile(yabs, q)
    candidates = np.where(yabs >= thr)[0]
    if candidates.size == 0:
        return _ref_from_max_peak(ppm_axis, y)
    idx = int(np.random.choice(candidates))
    return float(ppm_axis[idx])

# Funcion auxiliar de deformacion de fase, para cambio de fase desde un pico aleatorio mayoritario de todos los espectro

def choose_phase_strategy(ppm_axis, y):
    
# Funcion de deformacion de fase
 """Devuelve: (apply_phase(bool), ref_ppm(float or np.nan), mode_tag(str))según las proporciones pedidas"""

    # ——— Configuración de porcentajes de estrategia de cambio de fase (suman 1.0) ———
    PCT_NO_PHASE        = 1  # 30%
    PCT_MAX_PEAK        = 0  # 20%
    PCT_REF_PEAK        = 0  # 15%
    PCT_PROM_PEAK       = 0  # 20% 
    PCT_CENTER          = 0  # 15%
   
    u = random.random()
    cut_no   = PCT_NO_PHASE
    cut_max  = cut_no   + PCT_MAX_PEAK
    cut_ref  = cut_max  + PCT_REF_PEAK
    cut_prom = cut_ref  + PCT_PROM_PEAK
    
    if u < cut_no:
        return False, float('nan'), "no_phase"

    if u < cut_max:
        return True, _ref_from_max_peak(ppm_axis, y), "max_peak"

    if u < cut_ref:
        if REF_REFERENCE_PPM is not None:
            return True, float(REF_REFERENCE_PPM), "ref_peak"
        else:
            return True, _ppm_center(ppm_axis), "ref_peak_fallback_center"

    if u < cut_prom:





#def generadora_posta()








if __name__ == '__main__':


    ppm_axis = standardsSpec[0][:, 0]
    max_shift = 60

    root_dir = '/home/ray/'
    path_to_spectras_ind = os.path.join(root_dir, "Moleculas mol", "Espectros individuales")
   

    generador_spectros(path_to_spestros_ind)