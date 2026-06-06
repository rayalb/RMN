import numpy as np
import pandas as pd

import os

def read_spectrum_file(filepath):
    df = pd.read_csv(filepath, sep = "\t", header = None, name = ["ppm", "real", "imag"])
    x = df["ppm"].to_numpy()
    real = df["real"].to_numpy()
    return np.column_stack((x, real))

def load_individual_spectra(folder):
    filenames = [f for f in os.listdir(folder) if f.lower().endswith(".csv") and not f.startwith("~$")]
    spectra = []
    metadata = []

    for file in filenames:
        filepath = os.path.join(folder, file)
        data = read_spectrum_file(filepath)

        spectra.append(data)
        metadata.append({"filename": file, "points": data.shape[0], "has_image": True})

    return spectra, metadata, filenames

def cargar_perfil_metabolico(excel_path, sheet_name = "General"):
    return pd.read_excel(excel_path, sheet_name = sheet_name)

def normalize_names(s):
    import unicodedata
    s = os.path.basename(str(s))
    s = os.path.splitext(s)[0]
    s = s.lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.strip()

def merge_metadata_with_files(df, filenames):
    if "csv" in df.columns:
        filename_col = "csv"
    elif "filename_csv" in df.columns:
        filename_col = "filename_csv"
    else:
        filename_col = "filename"

    df = df.copy()
    df["filename_norm"] = df[filename_col].apply(normalize_names)

    df_loaded = pd.DataFrame({"filename_disk": filenames, "idx": range(len(filenames))})

    master_df = df.merge(df_loaded[["filename_norm", "filename_disk"]],
                         on = "filename_norm",how = "left", validate="m:1")

    return master_df

def diagnose_matching(master_df, filename_col):
    missing = master_df[master_df["idx"].isna()]
    print(f" Matcheadas: {len(master_df)-len(missing)}/{len(master_df)} filas")

    if len(missing):
        print("Filas sin match (muestra)")
        print(missing[[filename_col, "filename_norm"]].head())

def validate_ppm_axis(spectra, tol = 1e-6):
    x_ref = spectra[0][:, 0]

    for ii, spec in enumerate(spectra[1:], start=1):
        x_i = spec[:, 0]
        if not np.allclose(x_i, x_ref, rtol = 0, atol = tol):
            print(f"Eje ppm del espectro {ii} no coincide con el de referencia.")

def plot_combined_spectra(spectra, master_df, output_path=None):
    import matplotlib.pyplot as plt
    valid_idxs = master_df["idx"].dropna().astype(int).unique()
    idxs = sorted(valid_idxs)

    fig, ax = plt.subplots()
    for ii in idxs:
        x = spectra[ii][:, 0]
        y = spectra[ii][:, 1]

        name = master_df.loc[master_df["idx"] == ii, "Compuesto"].iloc[0]
        ax.plot(x, y, label = f"{name} (idx = {ii})")

    ax.invert_xaxis()
    ax.set_title("NMR Spectras")
    ax.set_xlabel("ppm")
    ax.set_ylabel{"Intensity"}
    ax.legend(fontsize = 8)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi = 300, bbox_inches = "tight")
        plt.close(fig)

# Plot Individual spectra
def safe_name(s, maxlen = 80):
    import re 
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:maxlen].strip("._-")

def plot_individual_spectra(spectra, master_df, out_dir=None):
    import matplotlib.pyplot as plt

    valid_idxs = master_df["idx"].dropna().astype(int).unique()
    idxs = sorted(valid_idxs)

    for ii in idxs:
        x = spectra[ii][:, 0]
        y = spectra[ii][:, 1]

        name = master_df.loc[master_df["idx"] == ii, "Compuesto"].iloc[0]
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.invert_xaxis()
        ax.set_xlabel("ppm")
        ax.set_ylabel("Intensity")
        ax.set_title(f"{name} (idx = {ii})")
        fig.tight_layout()
        if out_dir is not None:
            fname = f"{safe_name(name)}.pdf"
            path = os.path.join(out_dir, fname)
            fig.savefig(path, dpi = 300, bbox_inches="tight")
            plt.close(fig)

def definir_standard_interno(filepath):
    df = pd.read_csv(filepath, sep = "\t", header = None)
    v = df.iloc[:, 1].astype(float).to_numpy()
    return v

def main(root_dir):
    folder = os.path.join(root_dir, "Moleculas mol", "Espectros individuales")
    out_dir = os.path.join(root_dir, "Graficos basicos")
    os.makedirs(out_dir, exist_ok = True)

    # Load Spectra
    spectra, metadata, filenames = load_individual_spectra(folder)

    # Load Excel
    excel_path = os.path.join(root_dir, "perfil metabolico vino.xlsx")
    df = cargar_perfil_metabolico(excel_path)

    # Merge
    master_df = merge_metadata_with_files(df, filenames)

    # Diagnose
    diagnose_matching(master_df, "csv")

    # Validate axis
    validate_ppm_axis(spectra)

    # Plot
    plot_combined_spectra(spectra, master_df, output_path = os.path.join(out_dir, "espectros_individuales.pdf"))
    plot_individual_spectra(spectra, master_df, out_dir)

    # Internal standard
    std_path = os.path.join(folder, "metanol.csv")
    v_std = definir_standard_interno(std_path)

    return spectra, master_df, v_std