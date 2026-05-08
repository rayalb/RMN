import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generador_espectros import (load_individual_spectra, load_metabolic_profile, 
                                 load_internal_standard, match_spectra_to_metadata,
                                 _generate_single_mixture, normalize_name,
                                 )

from preprocesador_dataset import build_individual_cache, preprocess_single_mixture, load_metadata_individual



if __name__ == "__main__":
    root_dir  = "/home/ray/Documents/Quimica/"
    meta_ind_path = os.path.join(root_dir,"DataSet", 
                                 "metadata_espectros_individuales_con_TSP sodico(SI).xlsx")
    ind_dir = os.path.join(root_dir, "Espectros individuales + TSP sodico (SI)")
    EXCEL_FILENAME = "Perfil metabolico vino.xlsx"
    EXCEL_SHEET = "General"
    INTERNAL_STD_FILE = "tsp-d4 sodico.csv"
    INTERNAL_STD_FOLDER = "Moleculas mol"
    SPECTRA_SUBFOLDER = os.path.join(INTERNAL_STD_FOLDER, "Espectros individuales")
    SPECTRA_EXT = '.csv'

    PPM_IS = 0.0    # nominal ppm of iternal standard (TSP)
    WINDOWS_IS = 0.2 # +/- window around IS peak
    PPM_MIN = 0.0
    PPM_MAX = 10.0  # crop range


    MAX_SHIFT_PTS = 60
    DEFORMACION = True
    BASELINE_RANGE = (-0.02, 0.02)
    NOISE_LEVEL = 23.
    REF_REFERENCE_PPM = None
    PCT_NO_PHASE = 0.30
    PCT_MAX_PEAK = 0.20
    PCT_REF_PEAK = 0.15
    PCT_PROM_PEAK = 0.20
    PCT_CENTER = 0.15
    PHI0_RANGE = (-15.0, 15.0)
    PHI1_RANGE = (-3.0, 3.0)


    mixture = dict(
        aplicar_deformacion = DEFORMACION,
        REF_REFERENCE_PPM = REF_REFERENCE_PPM,
        PCT_NO_PHASE = PCT_NO_PHASE,
        PCT_MAX_PEAK = PCT_MAX_PEAK,
        PCT_REF_PEAK = PCT_REF_PEAK,
        PCT_PROM_PEAK = PCT_PROM_PEAK,
        PCT_CENTER = PCT_CENTER,
        phi0_range = PHI0_RANGE,
        phi1_range = PHI1_RANGE
    )

    standardSpec, standardsDictionary, x_ref = load_individual_spectra(
        root_dir = root_dir, subfolder = SPECTRA_SUBFOLDER, ext = SPECTRA_EXT
    )

    df = load_metabolic_profile(root_dir = root_dir, filename = EXCEL_FILENAME, sheet_name=EXCEL_SHEET)

    
    ppm_std, v_std = load_internal_standard(root_dir=root_dir, filename = INTERNAL_STD_FILE,
                                            subfolder=INTERNAL_STD_FOLDER)
    
    master_df, diagnostics = match_spectra_to_metadata(df = df,
                                                       standardsDictionary = standardsDictionary)
    ppm_axis = x_ref
   
    
    # load metadata individual
    meta_ind, comp_to_file = load_metadata_individual(meta_ind_path)
  
    # build individual cache
    
    ind_cache = build_individual_cache(meta_ind = meta_ind, ind_dir = ind_dir,
                                       comp_to_file = comp_to_file, ppm_is_nominal = PPM_IS,
                                        window_is = WINDOWS_IS, ppm_min = PPM_MIN, ppm_max = PPM_MAX)


    master_norm_to_real = {normalize_name(n): n for n in master_df["Compuesto"].tolist()}
    ind_to_master_idx = {}
    for comp_name in ind_cache.keys():
        norm = normalize_name(comp_name)
        if norm in master_norm_to_real:
            master_name = master_norm_to_real[norm]
            idx = master_df.loc[master_df["Compuesto"] == master_name, "idx"].values[0]
            ind_to_master_idx[comp_name] = int(idx)
   

    
    # build dataset
    idx_to_compname = {}
    for _, row in master_df.iterrows():
        idx_to_compname[int(row["idx"])] = row["Compuesto"]
    mix_complex, conc_dict, _, _ = _generate_single_mixture(mix_id = 0, master_df = master_df,
                                                            standardsSpec = standardSpec, ppm_axis = ppm_axis,
                                                            v = v_std, max_shift = MAX_SHIFT_PTS,
                                                            **mixture)
    
    mix_composition = {idx_to_compname[idx]: float(conc) for idx, conc in conc_dict.items()
                       if idx != "_IS" and idx in idx_to_compname}
    
    ppm_out, mix_prep = preprocess_single_mixture(mix_complex = mix_complex, ppm_axis = ppm_axis,
                                            ppm_is_nominal = PPM_IS, window_is = WINDOWS_IS,
                                            ppm_min = PPM_MIN, ppm_max= PPM_MAX, bin_factor = 1)
    
  
    R_list, S_list, y_list, comp_list = [], [], [], []

    for comp_name, ind_spec in ind_cache.items():
        if comp_name not in ind_to_master_idx:
            continue
        idx = ind_to_master_idx[comp_name]
        y_val = 1.0 if idx in conc_dict else 0.0
        R_list.append(ind_spec)
        S_list.append(mix_prep)
        y_list.append(y_val)
        comp_list.append(comp_name)

    R = np.stack(R_list, axis = 0).astype(np.float32)
    S = np.stack(S_list, axis = 0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)

    print(R.shape)
    print(S.shape)
    print(ppm_out.shape)

    for ii in range(R.shape[0]):
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ppm_out, R[ii], 'b')
        ax[0].grid(True)
        ax[1].plot(ppm_out, S[ii], 'r')
        ax[1].grid(True)

        plt.title(f'compuesto {comp_list[ii]}, esta: {y[ii]}')

        plt.tight_layout()
        plt.show()

    


           