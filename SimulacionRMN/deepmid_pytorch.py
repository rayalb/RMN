# =============================================================================
# deepmid_pytorch.py
# PyTorch implementation of the DeepMID architecture for NMR compound
# presence/absence classification from (R, S) spectrum pairs.
#
# Equivalent to the Keras DeepMID notebook but without contrastive loss.
# Compatible with VS Code and Google Colab.
# =============================================================================

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import json
import random
import warnings
from datetime import datetime

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


# =============================================================================
# SECTION 1 — Dataset
# =============================================================================

class NMRPairDataset(Dataset):
    """
    PyTorch Dataset for (R, S, y) NMR spectrum pairs.
    Wraps the aug_train/valid/test dicts from preprocesador_dataset.py.

    Parameters
    ----------
    aug : dict
        Output of build_full_dataset() or build_dataset_split().
        Must contain keys: 'R', 'S', 'y'.

    Returns per item
    ----------------
    R : torch.Tensor (float32, shape=(L,1))
        Individual compound spectrum (unsqueezed for Conv1d).
    S : torch.Tensor (float32, shape=(L,1))
        Mixture spectrum (unsqueezed for Conv1d).
    y : torch.Tensor (float32, shape=(1,))
        Binary label: 1 = present, 0 = absent.
    """

    def __init__(self, aug: dict):
        self.R = torch.tensor(aug["R"], dtype=torch.float32)   # (N, L)
        self.S = torch.tensor(aug["S"], dtype=torch.float32)   # (N, L)
        self.y = torch.tensor(aug["y"], dtype=torch.float32)   # (N, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # unsqueeze to (L, 1) for Conv1d input (channels last → will permute in model)
        return self.R[idx], self.S[idx], self.y[idx]


# =============================================================================
# SECTION 2 — Spatial Pyramid Pooling (1D adaptation)
# =============================================================================

class SpatialPyramidPooling2D(nn.Module):
    """
    2D Spatial Pyramid Pooling layer — exact equivalent of the Keras SPP.
    For each pool size n in pool_list, divides the (H, W) spatial map into
    n×n regions and applies max pooling, then concatenates all results.

    Output size = n_channels * sum(i*i for i in pool_list)
    e.g. pool_list=[1,2,3,4] → 1+4+9+16=30 → output = channels * 30

    Parameters
    ----------
    pool_list : list of int
        Number of bins per dimension per scale. Default: [1, 2, 3, 4].

    Input
    -----
    x : torch.Tensor (float32, shape=(batch, channels, H, W))

    Output
    ------
    torch.Tensor (float32, shape=(batch, channels * sum(i*i for i in pool_list)))

    Used by
    -------
    DeepMID
    """

    def __init__(self, pool_list: list = None):
        super().__init__()
        self.pool_list = pool_list or [1, 2, 3, 4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, H, W)
        batch_size = x.shape[0]
        outputs    = []

        for n in self.pool_list:
            # adaptive max pool to (n, n) spatial grid
            pooled = F.adaptive_max_pool2d(x, output_size=(n, n))  # (batch, C, n, n)
            outputs.append(pooled.reshape(batch_size, -1))          # (batch, C * n * n)

        return torch.cat(outputs, dim=1)   # (batch, C * sum(i*i for i in pool_list))


# =============================================================================
# SECTION 3 — DeepMID Architecture
# =============================================================================

class ConvBranch(nn.Module):
    """
    Shared convolutional branch applied independently to R and S inputs.
    Architecture: Conv1d(64, k=5) → ReLU → MaxPool(2) →
                  [Conv1d(64, k=5) → ReLU → (MaxPool every pool_every blocks)]

    Parameters
    ----------
    num_layers : int
        Number of additional conv blocks after the initial one. Default: 8.
    pool_every : int
        Apply MaxPool1d every this many blocks. Default: 2.

    Input
    -----
    x : torch.Tensor (float32, shape=(batch, 1, L))
        Spectrum signal with channel dimension added.

    Output
    ------
    torch.Tensor (float32, shape=(batch, 64, L'))
        Feature maps after convolutions and pooling.

    Used by
    -------
    DeepMID
    """

    def __init__(self, num_layers: int = 8, pool_every: int = 2):
        super().__init__()

        layers = []

        # Block 0: initial conv + relu + pool
        layers += [
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        ]

        # Blocks 1..num_layers
        for i in range(num_layers):
            layers += [
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.ReLU(),
            ]
            if (i + 1) % pool_every == 0:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.net = nn.Sequential(*layers)

        # Initialize Conv1d layers with he_normal (kaiming_normal) to match Keras
        for m in self.net:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMID(nn.Module):
    """
    Deep Mixture Identification network for NMR spectrum pair classification.

    Two independent convolutional branches process R (individual compound
    spectrum) and S (mixture spectrum). Their feature maps are concatenated
    along the channel axis and passed through a Conv2D → SPP → Dense pipeline
    to predict compound presence/absence.

    Architecture
    ------------
    input_R (batch, L) ──→ unsqueeze ──→ ConvBranch ──→ convR (batch, 64, L')
    input_S (batch, L) ──→ unsqueeze ──→ ConvBranch ──→ convS (batch, 64, L')
                                                              ↓
                                              cat([convR, convS], dim=1)
                                              → (batch, 128, L')
                                              → unsqueeze(2)
                                              → Conv2d(128, 128, k=(5,5), s=(2,2))
                                              → ReLU
                                              → SPP([1,2,3,4])
                                              → Dense(200) → ReLU → Dropout(0.2)
                                              → Dense(1) → Sigmoid
                                              → y_pred (batch, 1)

    Parameters
    ----------
    num_conv_layers : int
        Number of additional conv blocks in each branch. Default: 8.
    pool_every      : int
        MaxPool frequency in conv branches. Default: 2.
    pool_list       : list of int
        SPP bin sizes. Default: [1, 2, 3, 4].
    dropout         : float
        Dropout rate before final Dense. Default: 0.2.

    Input
    -----
    R : torch.Tensor (float32, shape=(batch, L))
    S : torch.Tensor (float32, shape=(batch, L))

    Output
    ------
    torch.Tensor (float32, shape=(batch, 1))
        Predicted probability of compound presence.

    Used by
    -------
    train_deepmid(), mc_dropout_predict()
    """

    def __init__(
        self,
        num_conv_layers: int  = 8,
        pool_every      : int  = 2,
        pool_list       : list = None,
        dropout         : float = 0.2,
    ):
        super().__init__()
        self.pool_list = pool_list or [1, 2, 3, 4]

        # Independent conv branches for R and S
        self.branch_R = ConvBranch(num_layers=num_conv_layers, pool_every=pool_every)
        self.branch_S = ConvBranch(num_layers=num_conv_layers, pool_every=pool_every)

        # Fusion conv (2D over merged feature maps)
        # Keras: Concatenate(axis=2) → (batch, L', 128) → expand → (batch, L', 128, 1)
        #        Conv2D(128, kernel=(5,5), stride=(2,2), padding='same')
        # PyTorch equivalent: input (batch, 1, L', 128) → Conv2d(1, 128, (5,5), (2,2), same)
        self.conv2d = nn.Conv2d(
            in_channels  = 1,
            out_channels = 128,
            kernel_size  = (5, 5),
            stride       = (2, 2),
            padding      = (2, 2),   # 'same' equivalent for k=5
        )

        # SPP 2D — matches Keras exactly
        # output size: 128 * sum(i*i for i in pool_list)
        # e.g. [1,2,3,4] → 128 * 30 = 3840
        self.spp     = SpatialPyramidPooling2D(pool_list=self.pool_list)
        spp_out_size = 128 * sum(i * i for i in self.pool_list)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(spp_out_size, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 1),
            nn.Sigmoid(),
        )

        # Initialize Conv2d with he_normal to match Keras
        nn.init.kaiming_normal_(self.conv2d.weight, mode="fan_in", nonlinearity="relu")
        if self.conv2d.bias is not None:
            nn.init.zeros_(self.conv2d.bias)

    def forward(self, R: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        # Add channel dim: (batch, L) → (batch, 1, L)
        R = R.unsqueeze(1)
        S = S.unsqueeze(1)

        # Conv branches: (batch, 1, L) → (batch, 64, L')
        convR = self.branch_R(R)
        convS = self.branch_S(S)

        # Match Keras: Concatenate(axis=2) on (batch, L', 64) tensors
        # → (batch, L', 128) then expand last dim → (batch, L', 128, 1)
        # PyTorch Conv2d expects (batch, C_in, H, W)
        # So: (batch, 1, L', 128) where H=L', W=128, C_in=1
        convR_t = convR.permute(0, 2, 1)              # (batch, L', 64)
        convS_t = convS.permute(0, 2, 1)              # (batch, L', 64)
        merged  = torch.cat([convR_t, convS_t], dim=2) # (batch, L', 128)
        merged  = merged.unsqueeze(1)                  # (batch, 1, L', 128)

        # Conv2D(128, kernel=(5,5), stride=(2,2), padding='same')
        merged = F.relu(self.conv2d(merged))             # (batch, 128, L'', 64)

        # SPP2D over (L'', 64) spatial dims: (batch, 128 * sum(i*i for i in pool_list))
        merged = self.spp(merged)

        # Classification head: (batch, 1)
        return self.classifier(merged)


# =============================================================================
# SECTION 4 — Training loop
# =============================================================================

def set_seeds(seed: int = 42):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed. Default: 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_deepmid(
    model          : DeepMID,
    aug_train      : dict,
    aug_valid      : dict,
    epochs         : int   = 50,
    batch_size     : int   = 32,
    lr             : float = 1e-4,
    patience       : int   = 8,
    min_delta      : float = 1e-3,
    save_path      : str   = None,
    device         : str   = None,
    num_workers    : int   = 0,
) -> dict:
    """
    Training loop for DeepMID with early stopping and model checkpointing.

    Parameters
    ----------
    model      : DeepMID
        Instantiated DeepMID model.
    aug_train  : dict
        Training split from build_full_dataset(). Keys: 'R', 'S', 'y'.
    aug_valid  : dict
        Validation split from build_full_dataset(). Keys: 'R', 'S', 'y'.
    epochs     : int
        Maximum number of training epochs. Default: 50.
    batch_size : int
        Batch size. Default: 32.
    lr         : float
        Adam learning rate. Default: 1e-4.
    patience   : int
        Early stopping patience (epochs without improvement). Default: 8.
    min_delta  : float
        Minimum accuracy improvement to reset patience. Default: 1e-3.
    save_path  : str or None
        Path to save best model weights (.pt file).
        If None, weights are not saved to disk.
    device     : str or None
        'cuda', 'cpu', or None (auto-detect). Default: None.
    num_workers: int
        DataLoader workers. Default: 0 (safe for Windows/Colab).

    Returns
    -------
    history : dict
        Training history with keys:
            - 'train_loss'     : list of float
            - 'train_accuracy' : list of float
            - 'val_loss'       : list of float
            - 'val_accuracy'   : list of float
            - 'best_epoch'     : int  (0-based)
            - 'best_val_acc'   : float

    Used by
    -------
    (training notebook)
    """
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")
    model = model.to(device)

    # Dataloaders
    train_loader = DataLoader(
        NMRPairDataset(aug_train),
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = (device == "cuda"),
    )
    valid_loader = DataLoader(
        NMRPairDataset(aug_valid),
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = (device == "cuda"),
    )

    optimizer  = Adam(model.parameters(), lr=lr)
    loss_fn    = nn.BCELoss()

    history = {
        "train_loss"    : [],
        "train_accuracy": [],
        "val_loss"      : [],
        "val_accuracy"  : [],
        "best_epoch"    : 0,
        "best_val_acc"  : 0.0,
    }

    best_val_acc   = 0.0
    best_weights   = None
    patience_count = 0

    print(f"Training on: {device.upper()}")
    print(f"Train batches/epoch: {len(train_loader)} | "
          f"Valid batches/epoch: {len(valid_loader)}")
    print("─" * 75)

    for epoch in range(epochs):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        train_bar = tqdm(
            train_loader,
            desc    = f"Epoch {epoch+1:>3}/{epochs} [Train]",
            leave   = False,
            ncols   = 80,
        )

        for R_batch, S_batch, y_batch in train_bar:
            R_batch = R_batch.to(device)
            S_batch = S_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(R_batch, S_batch)
            loss   = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            batch_loss     = loss.item()
            train_loss    += batch_loss * len(y_batch)
            predicted      = (y_pred >= 0.5).float()
            train_correct += (predicted == y_batch).sum().item()
            train_total   += len(y_batch)

            # live update inside the bar
            train_bar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "acc" : f"{train_correct/train_total:.4f}",
            })

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # ── Validation ───────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        val_bar = tqdm(
            valid_loader,
            desc  = f"Epoch {epoch+1:>3}/{epochs} [Valid]",
            leave = False,
            ncols = 80,
        )

        with torch.no_grad():
            for R_batch, S_batch, y_batch in val_bar:
                R_batch = R_batch.to(device)
                S_batch = S_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred   = model(R_batch, S_batch)
                loss     = loss_fn(y_pred, y_batch)

                val_loss    += loss.item() * len(y_batch)
                predicted    = (y_pred >= 0.5).float()
                val_correct += (predicted == y_batch).sum().item()
                val_total   += len(y_batch)

                val_bar.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "val_acc" : f"{val_correct/val_total:.4f}",
                })

        val_loss /= val_total
        val_acc   = val_correct / val_total
        elapsed   = time.time() - t0

        # ── Epoch summary ─────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        is_best = "  ✓ best" if val_acc > best_val_acc + min_delta else ""
        print(
            f"Epoch {epoch+1:>3}/{epochs} | "
            f"loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s{is_best}"
        )

        # ── Early stopping & checkpoint ───────────────────────────
        if val_acc > best_val_acc + min_delta:
            best_val_acc              = val_acc
            best_weights              = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"]     = epoch
            history["best_val_acc"]   = float(best_val_acc)
            patience_count            = 0

            if save_path is not None:
                torch.save(best_weights, save_path)
                print(f"  ✓ Best weights saved (val_acc={best_val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"\nBest epoch: {history['best_epoch']+1} | Best val_acc: {best_val_acc:.4f}")

    return history


# =============================================================================
# SECTION 5 — MC Dropout inference
# =============================================================================

def mc_dropout_predict(
    model      : DeepMID,
    aug        : dict,
    n_samples  : int   = 30,
    batch_size : int   = 64,
    device     : str   = None,
) -> tuple:
    """
    Monte Carlo Dropout inference for uncertainty estimation.
    Runs n_samples forward passes with dropout active and returns
    the mean and std of predicted probabilities.

    Parameters
    ----------
    model      : DeepMID
        Trained DeepMID model (must have Dropout layers).
    aug        : dict
        Dataset split dict with keys 'R' and 'S'.
    n_samples  : int
        Number of stochastic forward passes. Default: 30.
    batch_size : int
        Batch size for inference. Default: 64.
    device     : str or None
        'cuda', 'cpu', or None (auto-detect). Default: None.

    Returns
    -------
    mean_p : np.ndarray (float32, shape=(N,))
        Mean predicted probability across MC samples.
    std_p  : np.ndarray (float32, shape=(N,))
        Standard deviation of predicted probability (uncertainty).

    Used by
    -------
    (evaluation notebook)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.train()   # keep dropout active for MC sampling

    R = torch.tensor(aug["R"], dtype=torch.float32)
    S = torch.tensor(aug["S"], dtype=torch.float32)
    N = len(R)

    sum_p   = np.zeros(N, dtype=np.float64)
    sumsq_p = np.zeros(N, dtype=np.float64)

    with torch.no_grad():
        for t in range(n_samples):
            preds_t = np.zeros(N, dtype=np.float32)

            for start in range(0, N, batch_size):
                end      = min(start + batch_size, N)
                R_batch  = R[start:end].to(device)
                S_batch  = S[start:end].to(device)

                p = model(R_batch, S_batch).cpu().numpy().reshape(-1)
                preds_t[start:end] = p

            sum_p   += preds_t
            sumsq_p += preds_t ** 2

    mean_p = (sum_p / n_samples).astype(np.float32)
    var_p  = (sumsq_p / n_samples) - (mean_p.astype(np.float64) ** 2)
    var_p  = np.maximum(var_p, 0.0)
    std_p  = np.sqrt(var_p).astype(np.float32)

    model.eval()   # restore eval mode after MC sampling
    return mean_p, std_p


# =============================================================================
# SECTION 6 — Save / load run
# =============================================================================

def save_run(
    model        : DeepMID,
    history      : dict,
    aug_train    : dict,
    aug_valid    : dict,
    run_dir      : str,
    model_name   : str  = "deepmid",
    model_hparams: dict = None,
    train_hparams: dict = None,
    notes        : str  = "",
) -> str:
    """
    Saves model weights, training history, and metadata to disk.

    Saves:
        - {model_name}.best.pt       : best model state dict
        - info_modelo.json           : full run metadata + history

    Parameters
    ----------
    model         : DeepMID
        Trained model (with best weights already loaded).
    history       : dict
        Output of train_deepmid().
    aug_train     : dict
        Training split (used to log data shapes).
    aug_valid     : dict
        Validation split (used to log data shapes).
    run_dir       : str
        Directory where files will be saved (created if not exists).
    model_name    : str
        Base name for saved files. Default: "deepmid".
    model_hparams : dict or None
        Model hyperparameters to log. Default: None.
    train_hparams : dict or None
        Training hyperparameters to log. Default: None.
    notes         : str
        Optional notes to include in the JSON. Default: "".

    Returns
    -------
    str
        Path to the saved JSON file.

    Used by
    -------
    (training notebook)
    """
    os.makedirs(run_dir, exist_ok=True)

    weights_path = os.path.join(run_dir, f"{model_name}.best.pt")
    torch.save(model.state_dict(), weights_path)

    info = {
        "run": {
            "model_name" : model_name,
            "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes"      : notes,
        },
        "paths": {
            "run_dir"     : run_dir,
            "best_weights": weights_path,
        },
        "data": {
            "train_R_shape": list(aug_train["R"].shape),
            "train_S_shape": list(aug_train["S"].shape),
            "train_samples": int(aug_train["R"].shape[0]),
            "valid_R_shape": list(aug_valid["R"].shape),
            "valid_S_shape": list(aug_valid["S"].shape),
            "valid_samples": int(aug_valid["R"].shape[0]),
        },
        "model"   : {"hyperparameters": model_hparams or {}},
        "training": {"hyperparameters": train_hparams or {}},
        "results" : {
            "best_epoch"   : history["best_epoch"],
            "best_val_acc" : history["best_val_acc"],
            "epochs_trained": len(history["train_loss"]),
        },
        "history" : history,
    }

    json_path = os.path.join(run_dir, "info_modelo.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"✓ Weights saved : {weights_path}")
    print(f"✓ JSON saved    : {json_path}")
    return json_path


def load_run(
    run_dir        : str,
    model_hparams  : dict = None,
    device         : str  = None,
) -> tuple:
    """
    Loads a saved DeepMID model and its run metadata from disk.

    Parameters
    ----------
    run_dir       : str
        Directory containing info_modelo.json and .pt weights file.
    model_hparams : dict or None
        Override model hyperparameters. If None, reads from JSON.
    device        : str or None
        'cuda', 'cpu', or None (auto-detect). Default: None.

    Returns
    -------
    model : DeepMID
        Model with best weights loaded, in eval mode.
    info  : dict
        Full run metadata from info_modelo.json.

    Used by
    -------
    (evaluation notebook)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    json_path = os.path.join(run_dir, "info_modelo.json")
    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    hparams = model_hparams or info["model"].get("hyperparameters", {})

    model = DeepMID(
        num_conv_layers = hparams.get("num_conv_layers", 8),
        pool_every      = hparams.get("pool_every",      2),
        pool_list       = hparams.get("pool_list",       [1, 2, 3, 4]),
        dropout         = hparams.get("dropout",         0.2),
    )

    weights_path = info["paths"]["best_weights"]
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded from : {weights_path}")
    print(f"  Best epoch        : {info['results']['best_epoch'] + 1}")
    print(f"  Best val_acc      : {info['results']['best_val_acc']:.4f}")

    return model, info


# =============================================================================
# ENTRY POINT — edit parameters here and run
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))  # ensure local imports work

    from generador_espectros import (
        load_individual_spectra, load_metabolic_profile,
        load_internal_standard, match_spectra_to_metadata,
        normalize_name,
    )
    from preprocesador_dataset import (
        load_metadata_individual, build_individual_cache, build_full_dataset,
    )

    # ── General ──────────────────────────────────────────────────────────────
    ROOT_DIR     = r"C:\Users\tuusuario\Documents\NMR"   # <-- change this
    RESULTS_DIR  = os.path.join(ROOT_DIR, "Resultados", "Modelos")
    RANDOM_SEED  = 42
    set_seeds(RANDOM_SEED)

    # ── File paths ───────────────────────────────────────────────────────────
    META_IND_PATH = os.path.join(ROOT_DIR, "data set",
                    "metadata_espectros_individuales_con_TSP sodico(SI).xlsx")
    IND_DIR       = os.path.join(ROOT_DIR, "Espectros individuales + TSP sodico (SI)")
    EXCEL_FILENAME      = "perfil metabolico vino.xlsx"
    EXCEL_SHEET         = "General"
    INTERNAL_STD_FILE   = "tsp-d4 sodico.csv"
    INTERNAL_STD_FOLDER = "Moleculas mol"
    SPECTRA_SUBFOLDER   = os.path.join("Moleculas mol", "Espectros individuales")

    # ── Model hyperparameters ────────────────────────────────────────────────
    NUM_CONV_LAYERS = 8
    POOL_EVERY      = 2
    POOL_LIST       = [1, 2, 3, 4]
    DROPOUT         = 0.2

    # ── Training hyperparameters ─────────────────────────────────────────────
    EPOCHS     = 50
    BATCH_SIZE = 32
    LR         = 1e-4
    PATIENCE   = 8
    MIN_DELTA  = 1e-3

    # ── Dataset ──────────────────────────────────────────────────────────────
    N_TOTAL    = 2000
    TRAIN_FRAC = 0.70
    VALID_FRAC = 0.15

    # ── Load pipeline data ───────────────────────────────────────────────────
    standardsSpec, standardsDictionary, x_ref = load_individual_spectra(
        root_dir=ROOT_DIR, subfolder=SPECTRA_SUBFOLDER)
    df             = load_metabolic_profile(root_dir=ROOT_DIR, filename=EXCEL_FILENAME,
                                            sheet_name=EXCEL_SHEET)
    ppm_std, v_std = load_internal_standard(root_dir=ROOT_DIR,
                                            filename=INTERNAL_STD_FILE,
                                            subfolder=INTERNAL_STD_FOLDER)
    master_df, _   = match_spectra_to_metadata(df=df, standardsDictionary=standardsDictionary)
    ppm_axis       = x_ref

    meta_ind, comp_to_file = load_metadata_individual(META_IND_PATH)
    ind_cache = build_individual_cache(meta_ind=meta_ind, ind_dir=IND_DIR,
                                       comp_to_file=comp_to_file)

    master_norm_to_real = {normalize_name(n): n for n in master_df["Compuesto"].tolist()}
    ind_to_master_idx   = {}
    for comp_name in ind_cache.keys():
        norm = normalize_name(comp_name)
        if norm in master_norm_to_real:
            master_name = master_norm_to_real[norm]
            idx = master_df.loc[master_df["Compuesto"] == master_name, "idx"].values[0]
            ind_to_master_idx[comp_name] = int(idx)

    # ── Generate datasets on-the-fly ─────────────────────────────────────────
    aug_train, aug_valid, aug_test = build_full_dataset(
        master_df         = master_df,
        standardsSpec     = standardsSpec,
        ppm_axis          = ppm_axis,
        v                 = v_std,
        ind_cache         = ind_cache,
        ind_to_master_idx = ind_to_master_idx,
        n_total           = N_TOTAL,
        train_frac        = TRAIN_FRAC,
        valid_frac        = VALID_FRAC,
        random_seed       = RANDOM_SEED,
        mixture_kwargs    = dict(
            baseline_range      = (-0.02, 0.02),
            noise_level         = 0.02,
            aplicar_deformacion = False,
            REF_REFERENCE_PPM   = None,
            PCT_NO_PHASE        = 0.30,
            PCT_MAX_PEAK        = 0.20,
            PCT_REF_PEAK        = 0.15,
            PCT_PROM_PEAK       = 0.20,
            PCT_CENTER          = 0.15,
            phi0_range          = (-15.0, 15.0),
            phi1_range          = (-3.0,   3.0),
        )
    )

    print(f"Train: {aug_train['R'].shape} {aug_train['S'].shape} {aug_train['y'].shape}")
    print(f"Valid: {aug_valid['R'].shape} {aug_valid['S'].shape} {aug_valid['y'].shape}")
    print(f"Test : {aug_test['R'].shape}  {aug_test['S'].shape}  {aug_test['y'].shape}")

    # ── Build and train model ─────────────────────────────────────────────────
    model = DeepMID(
        num_conv_layers = NUM_CONV_LAYERS,
        pool_every      = POOL_EVERY,
        pool_list       = POOL_LIST,
        dropout         = DROPOUT,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(RESULTS_DIR, f"deepmid_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    history = train_deepmid(
        model      = model,
        aug_train  = aug_train,
        aug_valid  = aug_valid,
        epochs     = EPOCHS,
        batch_size = BATCH_SIZE,
        lr         = LR,
        patience   = PATIENCE,
        min_delta  = MIN_DELTA,
        save_path  = os.path.join(run_dir, "deepmid.best.pt"),
    )

    # ── Save run ──────────────────────────────────────────────────────────────
    save_run(
        model         = model,
        history       = history,
        aug_train     = aug_train,
        aug_valid     = aug_valid,
        run_dir       = run_dir,
        model_name    = "deepmid",
        model_hparams = {
            "num_conv_layers": NUM_CONV_LAYERS,
            "pool_every"     : POOL_EVERY,
            "pool_list"      : POOL_LIST,
            "dropout"        : DROPOUT,
        },
        train_hparams = {
            "epochs"    : EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr"        : LR,
            "patience"  : PATIENCE,
            "min_delta" : MIN_DELTA,
            "n_total"   : N_TOTAL,
        },
    )

    # ── MC Dropout on test set ────────────────────────────────────────────────
    mean_p, std_p = mc_dropout_predict(
        model      = model,
        aug        = aug_test,
        n_samples  = 30,
        batch_size = 64,
    )

    print(f"\nTest MC Dropout — mean_p range: [{mean_p.min():.3f}, {mean_p.max():.3f}]")
    print(f"Test MC Dropout — std_p  range: [{std_p.min():.3f},  {std_p.max():.3f}]")
