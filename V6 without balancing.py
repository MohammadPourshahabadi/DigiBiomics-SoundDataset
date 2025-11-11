# -*- coding: utf-8 -*-
"""
ICBHI 2017 cycle classification (from scratch, no pretrain) with:

- Group-wise splits (no patient leakage)
    * ~15% test by subject IDs
    * From remaining: ~85/15 train/val by subject
- No oversampling
- Class-weighted focal loss on original train distribution
- Waveform augmentations (gain, noise, shift) on-the-fly for training
- Early stopping (patience=10) on val macro-F1
- Mixed precision (if GPU available)
- Full metrics & plots:
    * per-epoch metrics CSV
    * per-class reports (val)
    * confusion matrices (val & test, incl. one-vs-rest)
    * ROC + AUC (val & test, per-class + micro + macro)
    * learning curves
    * hyperparameters table
    * test metrics summary + per-class report
"""

import os, glob, random, math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset

# ------------------- PATHS ------------------- #
BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/Datasets/ICBHI 2017 Challenge/ICBHI_final_database"

candidates = glob.glob(os.path.join(BASE_PATH, "**/*.wav"), recursive=True)
if len(candidates) == 0:
    raise FileNotFoundError(f"No .wav files found under {BASE_PATH}.")
else:
    DATASET_ROOT = os.path.commonpath(candidates)
    print(f"[Auto-detect] Using DATASET_ROOT = {DATASET_ROOT}")

OUTDIR = "/content/drive/MyDrive/Colab Notebooks/Datasets/Results_ICBHI_Scratch_Focal_NoOversample"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------- HYPERPARAMS ------------------- #
SEED = 1337
EPOCHS = 45
BATCH_TRAIN = 16
BATCH_VAL = 32
LR = 1.5e-4
WEIGHT_DECAY = 1e-3
TARGET_SEC = 2.5
PATIENCE = 10              # early stopping patience

SR = 16000
WIN = int(0.025 * SR)      # 25 ms
HOP = int(0.010 * SR)      # 10 ms
N_MELS = 40
FMIN = 20
FMAX = 8000
LABELS = ["normal", "crackle", "wheeze", "both"]
N_CLASSES = len(LABELS)

# ------------------- UTILS ------------------- #
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cycle_label(c, w):
    if c == 0 and w == 0: return 0
    if c == 1 and w == 0: return 1
    if c == 0 and w == 1: return 2
    return 3

def subject_id_of(wav_path: str):
    """Extract subject/patient id from filename like '221_2b1_...wav' -> '221'."""
    base = os.path.basename(wav_path)
    return base.split('_')[0]

skipped_files = []

# ------------------- LOADER ------------------- #
def load_wav_16k(path, target_sr=16000):
    import soundfile as sf
    try:
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data).unsqueeze(0)
    except Exception:
        try:
            import librosa
            data, sr = librosa.load(path, sr=None, mono=True)
            wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        except Exception:
            try:
                wav, sr = torchaudio.load(path)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
            except Exception as e:
                print(f"[Warning] Skipping file {path}: {e}")
                skipped_files.append(path)
                return None
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav

# ------------------- AUGMENTATION ------------------- #
def rand_bool(p): 
    return random.random() < p

def add_gaussian_noise(wav, snr_db_range=(12, 25)):
    x = wav.squeeze(0)
    rms = torch.sqrt(torch.mean(x ** 2) + 1e-12)
    snr_db = random.uniform(*snr_db_range)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_rms = rms / math.sqrt(snr_lin)
    noise = torch.randn_like(x) * noise_rms
    return (x + noise).unsqueeze(0)

def random_gain(wav, db_range=(-4, 4)):
    gain_db = random.uniform(*db_range)
    gain = 10.0 ** (gain_db / 20.0)
    return wav * gain

def time_shift(wav, max_shift_sec=0.2, sr=16000):
    max_shift = int(max_shift_sec * sr)
    if max_shift <= 0:
        return wav
    shift = random.randint(-max_shift, max_shift)
    x = wav.squeeze(0)
    if shift > 0:
        y = torch.cat([x[-shift:], x[:-shift]], dim=0)
    elif shift < 0:
        shift = -shift
        y = torch.cat([x[shift:], x[:shift]], dim=0)
    else:
        y = x
    return y.unsqueeze(0)

def apply_random_augment(wav):
    if rand_bool(0.7):
        wav = random_gain(wav, (-4, 4))
    if rand_bool(0.6):
        wav = add_gaussian_noise(wav, (12, 25))
    if rand_bool(0.5):
        wav = time_shift(wav, 0.2, SR)
    return wav

# ------------------- DATASET ------------------- #
@dataclass
class CycleItem:
    wav_path: str
    start_s: float
    end_s: float
    y: int

class ICBHICycleDataset(Dataset):
    def __init__(self, root, target_sec=2.5, random_crop=True, augment=False):
        self.root = root
        self.target_sec = target_sec
        self.random_crop = random_crop
        self.augment = augment
        self.items = []
        self._index()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=1024,
            win_length=WIN,
            hop_length=HOP,
            f_min=FMIN,
            f_max=FMAX,
            n_mels=N_MELS,
            center=True,
            power=2.0
        )
        self.amp2db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def _index(self):
        txts = sorted(glob.glob(os.path.join(self.root, "**/*.txt"), recursive=True))
        wavs = sorted(glob.glob(os.path.join(self.root, "**/*.wav"), recursive=True))
        wav_map = {os.path.basename(w).lower(): w for w in wavs}
        for txt in txts:
            cand_wav = txt[:-4] + ".wav"
            if not os.path.isfile(cand_wav):
                cand_wav = wav_map.get(os.path.basename(cand_wav).lower())
                if not cand_wav:
                    continue
            with open(txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    try:
                        s, e = float(parts[0]), float(parts[1])
                        c, w = int(float(parts[2])), int(float(parts[3]))
                        if e > s:
                            self.items.append(
                                CycleItem(cand_wav, s, e, cycle_label(c, w))
                            )
                    except:
                        continue
        print(f"[Dataset] Found {len(self.items)} cycles.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        wav = load_wav_16k(it.wav_path)
        if wav is None:
            return None

        start, end = int(it.start_s * SR), int(it.end_s * SR)
        seg = wav[:, start:end]
        target_len = int(self.target_sec * SR)

        if seg.shape[-1] == 0:
            seg = torch.zeros(1, target_len)
        if seg.shape[-1] < target_len:
            seg = nn.functional.pad(seg, (0, target_len - seg.shape[-1]))
        elif seg.shape[-1] > target_len:
            if self.random_crop:
                off = np.random.randint(0, seg.shape[-1] - target_len + 1)
            else:
                off = (seg.shape[-1] - target_len) // 2
            seg = seg[:, off:off + target_len]

        if self.augment:
            seg = apply_random_augment(seg)

        S = self.amp2db(self.melspec(seg) + 1e-10)
        S = (S - S.mean()) / (S.std() + 1e-6)
        return S, it.y

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None
    X = torch.stack([b[0] for b in batch])
    y = torch.tensor([b[1] for b in batch])
    return X, y

# ------------------- MODEL ------------------- #
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ScratchFeatureExtractor(nn.Module):
    def __init__(self, n_mels=40):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.Dropout(0.1),
        )
        conv_dim = (n_mels // 4) * 128
        self.rnn = nn.LSTM(
            conv_dim,
            256,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
    def forward(self, x):
        B, _, M, T = x.shape
        h = self.features(x)              # (B, C, M', T')
        h = h.permute(0, 3, 1, 2).contiguous().view(B, h.shape[3], -1)
        y, _ = self.rnn(h)               # (B, T', 512)
        return y

class ICBHIClassifier(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.enc = ScratchFeatureExtractor(N_MELS)
        self.attn = nn.Linear(512, 1)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, n_classes))
    def forward(self, x):
        seq = self.enc(x)
        w = torch.softmax(self.attn(seq).squeeze(-1), dim=1)
        emb = (seq * w.unsqueeze(-1)).sum(1)
        return self.head(emb)

# ------------------- FOCAL LOSS ------------------- #
class FocalLoss(nn.Module):
    """
    Multi-class focal loss with optional class weights.
    gamma: focusing parameter (2.0 common)
    weight: tensor [C] of class weights.
    """
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: (B, C), target: (B,)
        log_prob = torch.log_softmax(logits, dim=1)
        prob = torch.softmax(logits, dim=1)

        idx = torch.arange(target.size(0), device=logits.device)
        pt = prob[idx, target]                # p_t
        focal_factor = (1 - pt) ** self.gamma

        ce = -log_prob[idx, target]           # cross-entropy per sample

        if self.weight is not None:
            w = self.weight.to(logits.device)[target]
            ce = ce * w

        loss = focal_factor * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# ------------------- GROUP-WISE SPLIT ------------------- #
def groupwise_split_indices(base_dataset, test_size=0.15, val_size_within_remain=0.15, seed=SEED):
    items = base_dataset.items
    groups = np.array([subject_id_of(ci.wav_path) for ci in items])
    all_idx = np.arange(len(items))

    # Test split
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(all_idx, groups=groups))

    # Val split from remaining
    groups_remain = groups[train_val_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_within_remain, random_state=seed)
    train_idx_rel, val_idx_rel = next(gss_val.split(train_val_idx, groups=groups_remain))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    N = len(all_idx)
    print(f"[Group Split] N={N}  "
          f"train={len(train_idx)} ({len(train_idx)/N:.3%})  "
          f"val={len(val_idx)} ({len(val_idx)/N:.3%})  "
          f"test={len(test_idx)} ({len(test_idx)/N:.3%})")
    return train_idx, val_idx, test_idx

# ------------------- SAVE HELPERS ------------------- #
def save_hyperparams():
    H = {
        "SEED": SEED,
        "EPOCHS": EPOCHS,
        "BATCH_TRAIN": BATCH_TRAIN,
        "BATCH_VAL": BATCH_VAL,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "TARGET_SEC": TARGET_SEC,
        "PATIENCE": PATIENCE,
        "SR": SR,
        "WIN": WIN,
        "HOP": HOP,
        "N_MELS": N_MELS,
        "FMIN": FMIN,
        "FMAX": FMAX,
        "LABELS": ",".join(LABELS),
        "LOSS": "FocalLoss(gamma=2.0, class-weighted)",
        "OVERSAMPLING": "None (group-wise splits only)",
    }
    df = pd.DataFrame({"hyperparameter": list(H.keys()),
                       "value": list(H.values())})
    df.to_csv(os.path.join(OUTDIR, "hyperparameters.csv"), index=False)

def save_confusions(y_true, y_pred, labels, outdir, prefix):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(os.path.join(outdir, f"{prefix}_confusion_matrix.csv"))

    plt.figure(figsize=(6,5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix ({prefix})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_confusion_matrix.png"), dpi=150)
    plt.close()

    # one-vs-rest per class
    for i, name in enumerate(labels):
        yt_bin = (y_true == i).astype(int)
        yp_bin = (y_pred == i).astype(int)
        cm_i = confusion_matrix(yt_bin, yp_bin, labels=[0,1])
        df_i = pd.DataFrame(
            cm_i,
            index=[f"Actual not {name}", f"Actual {name}"],
            columns=[f"Pred not {name}", f"Pred {name}"],
        )
        df_i.to_csv(os.path.join(outdir, f"{prefix}_confusion_{name}_onevsrest.csv"))

        plt.figure(figsize=(4.2,3.6))
        sns.heatmap(df_i, annot=True, fmt="d", cmap="Purples")
        plt.title(f"{name}: One-vs-Rest ({prefix})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{prefix}_confusion_{name}_onevsrest.png"), dpi=150)
        plt.close()

def save_multiclass_roc(y_true, y_prob, labels, outdir, prefix):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim != 2 or y_prob.shape[1] != len(labels) or len(y_prob) == 0:
        print(f"[Warn] ROC skipped for {prefix} (no valid probabilities).")
        return

    Y_true = label_binarize(y_true, classes=list(range(len(labels))))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_true.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(labels))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(labels)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(labels)
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # save AUC table
    rows = [{"class": labels[i], "AUC": roc_auc[i]} for i in range(len(labels))]
    rows += [{"class": "micro", "AUC": roc_auc["micro"]},
             {"class": "macro", "AUC": roc_auc["macro"]}]
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{prefix}_auc.csv"), index=False)

    # plot ROC curves
    plt.figure(figsize=(7,6))
    for i, name in enumerate(labels):
        plt.plot(fpr[i], tpr[i], lw=1.6, label=f"{name} (AUC={roc_auc[i]:.3f})")
    plt.plot(fpr["micro"], tpr["micro"], lw=2.0, linestyle="--", label=f"micro (AUC={roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], lw=2.0, linestyle="-.", label=f"macro (AUC={roc_auc['macro']:.3f})")
    plt.plot([0,1],[0,1],"k:", lw=1)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({prefix})")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_roc.png"), dpi=160)
    plt.close()

# ------------------- MAIN ------------------- #
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Main] Using device:", device)

    save_hyperparams()

    # Base dataset for splits
    base = ICBHICycleDataset(DATASET_ROOT, TARGET_SEC, random_crop=True, augment=False)

    # Group-wise split
    train_idx, val_idx, test_idx = groupwise_split_indices(base)

    # Class weights from train
    train_labels_orig = np.array([base.items[i].y for i in train_idx])
    class_counts = np.bincount(train_labels_orig, minlength=N_CLASSES)
    print("Train class counts:", class_counts)

    class_weights = class_counts.sum() / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print("Class weights:", class_weights)

    # Datasets / loaders
    train_ds = Subset(ICBHICycleDataset(DATASET_ROOT, TARGET_SEC, True,  augment=True),  train_idx)
    val_ds   = Subset(ICBHICycleDataset(DATASET_ROOT, TARGET_SEC, False, augment=False), val_idx.tolist())
    test_ds  = Subset(ICBHICycleDataset(DATASET_ROOT, TARGET_SEC, False, augment=False), test_idx.tolist())

    train_dl = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,  collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_VAL,   shuffle=False, collate_fn=collate_fn)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_VAL,   shuffle=False, collate_fn=collate_fn)

    # Model / optimizer / loss
    model = ICBHIClassifier(N_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = FocalLoss(gamma=2.0, weight=class_weights_tensor, reduction="mean")
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    history = []
    best_mf1 = -1.0
    no_improve = 0

    # ---------- Training loop ----------
    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        model.train()
        yt, yp = [], []

        for X, y in tqdm(train_dl, desc=f"Training epoch {epoch}", leave=False):
            if X is None:
                continue
            X, y = X.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(X)
                loss = crit(out, y)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            yp.extend(out.argmax(1).detach().cpu().numpy())
            yt.extend(y.detach().cpu().numpy())

        tr_acc = accuracy_score(yt, yp)
        tr_mf1 = f1_score(yt, yp, average="macro")

        # ----- Validation -----
        model.eval()
        yt_val, yp_val = [], []
        yprob_val_list = []

        with torch.no_grad():
            for X, y in tqdm(val_dl, desc=f"Validation epoch {epoch}", leave=False):
                if X is None:
                    continue
                X, y = X.to(device), y.to(device)
                out = model(X)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                yprob_val_list.append(probs)
                yp_val.extend(out.argmax(1).cpu().numpy())
                yt_val.extend(y.cpu().numpy())

        yprob_val = np.vstack(yprob_val_list) if yprob_val_list else np.zeros((0, N_CLASSES))
        va_acc = accuracy_score(yt_val, yp_val)
        va_mf1 = f1_score(yt_val, yp_val, average="macro")

        metrics = {
            "epoch": epoch,
            "train_acc": tr_acc,
            "val_acc": va_acc,
            "train_mf1": tr_mf1,
            "val_mf1": va_mf1,
            "val_precision_micro": precision_score(yt_val, yp_val, average="micro", zero_division=0),
            "val_recall_micro": recall_score(yt_val, yp_val, average="micro", zero_division=0),
            "val_f1_micro": f1_score(yt_val, yp_val, average="micro", zero_division=0),
            "val_precision_macro": precision_score(yt_val, yp_val, average="macro", zero_division=0),
            "val_recall_macro": recall_score(yt_val, yp_val, average="macro", zero_division=0),
            "val_f1_macro": f1_score(yt_val, yp_val, average="macro", zero_division=0),
        }
        history.append(metrics)

        # per-class report on val
        rep = classification_report(yt_val, yp_val, target_names=LABELS, output_dict=True, zero_division=0)
        df_rep = pd.DataFrame(rep).T
        df_rep["epoch"] = epoch
        df_rep.to_csv(os.path.join(OUTDIR, f"metrics_per_class_epoch{epoch:03d}.csv"))

        print(f"[Epoch {epoch}] Train Acc={tr_acc:.3f} Val Acc={va_acc:.3f} Val F1={va_mf1:.3f}")

        # early stopping on val macro-F1
        if va_mf1 > best_mf1:
            best_mf1 = va_mf1
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(OUTDIR, "best.ckpt"))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs → early stop.")
                break

    # ---------- Save learning curves ----------
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(OUTDIR, "metrics_per_epoch.csv"), index=False)

    if not hist_df.empty:
        plt.figure(figsize=(7,5))
        plt.plot(hist_df["epoch"], hist_df["train_mf1"], label="Train F1", marker="o")
        plt.plot(hist_df["epoch"], hist_df["val_mf1"], label="Val F1", marker="o")
        plt.plot(hist_df["epoch"], hist_df["val_acc"], label="Val Acc", marker="x")
        plt.xlabel("Epoch"); plt.ylabel("Score")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "learning_curves.png"), dpi=150)
        plt.close()

    # ---------- Evaluate best model on VAL & TEST ----------
    best_path = os.path.join(OUTDIR, "best.ckpt")
    if os.path.isfile(best_path):
        best_model = ICBHIClassifier(N_CLASSES).to(device)
        best_model.load_state_dict(torch.load(best_path, map_location=device))
        best_model.eval()

        # VAL
        yt_val_best, yp_val_best, yprob_val_best = [], [], []
        with torch.no_grad():
            for X, y in tqdm(val_dl, desc="Best model eval (val set)"):
                if X is None:
                    continue
                X = X.to(device)
                out = best_model(X)
                yprob_val_best.append(torch.softmax(out, dim=1).cpu().numpy())
                yp_val_best.extend(out.argmax(1).cpu().numpy())
                yt_val_best.extend(y.numpy())
        yprob_val_best = np.vstack(yprob_val_best) if yprob_val_best else np.zeros((0, N_CLASSES))
        save_confusions(yt_val_best, yp_val_best, LABELS, OUTDIR, prefix="best_val")
        if len(yprob_val_best):
            save_multiclass_roc(yt_val_best, yprob_val_best, LABELS, OUTDIR, prefix="best_val")

        # TEST
        yt_test, yp_test, yprob_test = [], [], []
        with torch.no_grad():
            for X, y in tqdm(test_dl, desc="Best model eval (test set)"):
                if X is None:
                    continue
                X = X.to(device)
                out = best_model(X)
                yprob_test.append(torch.softmax(out, dim=1).cpu().numpy())
                yp_test.extend(out.argmax(1).cpu().numpy())
                yt_test.extend(y.numpy())
        yprob_test = np.vstack(yprob_test) if yprob_test else np.zeros((0, N_CLASSES))

        save_confusions(yt_test, yp_test, LABELS, OUTDIR, prefix="best_test")
        if len(yprob_test):
            save_multiclass_roc(yt_test, yprob_test, LABELS, OUTDIR, prefix="best_test")

        # Test metrics summary + per-class
        test_metrics = {
            "test_accuracy": accuracy_score(yt_test, yp_test),
            "test_precision_micro": precision_score(yt_test, yp_test, average="micro", zero_division=0),
            "test_recall_micro": recall_score(yt_test, yp_test, average="micro", zero_division=0),
            "test_f1_micro": f1_score(yt_test, yp_test, average="micro", zero_division=0),
            "test_precision_macro": precision_score(yt_test, yp_test, average="macro", zero_division=0),
            "test_recall_macro": recall_score(yt_test, yp_test, average="macro", zero_division=0),
            "test_f1_macro": f1_score(yt_test, yp_test, average="macro", zero_division=0),
            "test_balanced_accuracy": balanced_accuracy_score(yt_test, yp_test),
            "test_kappa": cohen_kappa_score(yt_test, yp_test),
            "test_mcc": matthews_corrcoef(yt_test, yp_test),
        }
        pd.DataFrame([test_metrics]).to_csv(
            os.path.join(OUTDIR, "best_test_metrics_summary.csv"),
            index=False,
        )

        test_report = classification_report(
            yt_test, yp_test, target_names=LABELS, output_dict=True, zero_division=0
        )
        pd.DataFrame(test_report).T.to_csv(
            os.path.join(OUTDIR, "best_test_classification_report.csv")
        )

        print(
            "[Best/Test]"
            f" Acc={test_metrics['test_accuracy']:.3f}"
            f" F1_macro={test_metrics['test_f1_macro']:.3f}"
            f" F1_micro={test_metrics['test_f1_micro']:.3f}"
            f" BalAcc={test_metrics['test_balanced_accuracy']:.3f}"
            f" κ={test_metrics['test_kappa']:.3f}"
            f" MCC={test_metrics['test_mcc']:.3f}"
        )

if __name__ == "__main__":
    main()
