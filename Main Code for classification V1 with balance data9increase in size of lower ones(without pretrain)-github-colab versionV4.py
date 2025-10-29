# -*- coding: utf-8 -*-
"""
ICBHI 2017 cycle classification (from scratch, no pretrain) with group-wise splits.
Now with:
- Group-wise Test split (~15% by samples, no patient leakage), via GroupShuffleSplit on subject IDs
- From remaining groups: group-wise Train/Val = ~85%/15%
- Training set balancing by oversampling (indices only; aug on-the-fly)
- Light waveform augmentations (gain, noise, shift)
- Early stopping (patience=10)
- Mixed precision (if GPU available)
- Full metrics (macro/micro precision/recall/F1), per-class reports each epoch
- Confusion matrices (overall + one-vs-rest), ROC + AUC (per-class, micro, macro)
- Learning curves and hyperparameters table
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
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset, DataLoader, Subset

# ------------------- PATHS ------------------- #
BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/Datasets/ICBHI 2017 Challenge/ICBHI_final_database"

candidates = glob.glob(os.path.join(BASE_PATH, "**/*.wav"), recursive=True)
if len(candidates) == 0:
    raise FileNotFoundError(f"No .wav files found under {BASE_PATH}.")
else:
    DATASET_ROOT = os.path.commonpath(candidates)
    print(f"[Auto-detect] Using DATASET_ROOT = {DATASET_ROOT}")

OUTDIR = "/content/drive/MyDrive/Colab Notebooks/Datasets/Results_ICBHI_Scratch_FullMetrics"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------- HYPERPARAMS ------------------- #
SEED = 1337
EPOCHS = 30
BATCH_TRAIN = 16
BATCH_VAL = 32
LR = 1.5e-4
WEIGHT_DECAY = 1e-3
TARGET_SEC = 2.5
BALANCE_DATA = True          # oversample TRAIN ONLY to equalize class counts
PATIENCE = 10                # early stopping patience

SR = 16000
WIN = int(0.025 * SR)        # 25 ms
HOP = int(0.010 * SR)        # 10 ms
N_MELS = 40
FMIN = 20
FMAX = 8000
LABELS = ["normal", "crackle", "wheeze", "both"]
N_CLASSES = len(LABELS)

# Quick peek (optional)
test_path = "/content/drive/MyDrive/Colab Notebooks/Datasets/ICBHI 2017 Challenge"
print("Sample .wav files found:")
for f in glob.glob(os.path.join(test_path, "**/*.wav"), recursive=True)[:5]:
    print("  ", f)
print("Sample .txt files found:")
for f in glob.glob(os.path.join(test_path, "**/*.txt"), recursive=True)[:5]:
    print("  ", f)

# ------------------- UTILS ------------------- #
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def cycle_label(c, w):
    if c == 0 and w == 0: return 0
    if c == 1 and w == 0: return 1
    if c == 0 and w == 1: return 2
    return 3

def subject_id_of(wav_path: str):
    """Extract subject/patient id from filename like '221_2b1_...wav' -> '221'."""
    base = os.path.basename(wav_path)
    first = base.split('_')[0]
    return first

skipped_files = []

# ------------------- ROBUST LOADER ------------------- #
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

# ------------------- AUGMENTATIONS ------------------- #
def rand_bool(p): return random.random() < p

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
    if max_shift <= 0: return wav
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
    if rand_bool(0.7): wav = random_gain(wav, (-4, 4))
    if rand_bool(0.6): wav = add_gaussian_noise(wav, (12, 25))
    if rand_bool(0.5): wav = time_shift(wav, 0.2, SR)
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
            sample_rate=SR, n_fft=1024, win_length=WIN, hop_length=HOP,
            f_min=FMIN, f_max=FMAX, n_mels=N_MELS, center=True, power=2.0)
        self.amp2db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def _index(self):
        txts = sorted(glob.glob(os.path.join(self.root, "**/*.txt"), recursive=True))
        wavs = sorted(glob.glob(os.path.join(self.root, "**/*.wav"), recursive=True))
        wav_map = {os.path.basename(w).lower(): w for w in wavs}
        for txt in txts:
            cand_wav = txt[:-4] + ".wav"
            if not os.path.isfile(cand_wav):
                cand_wav = wav_map.get(os.path.basename(cand_wav).lower())
                if not cand_wav: continue
            with open(txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 4: continue
                    try:
                        s, e = float(parts[0]), float(parts[1])
                        c, w = int(float(parts[2])), int(float(parts[3]))
                        if e > s:
                            self.items.append(CycleItem(cand_wav, s, e, cycle_label(c, w)))
                    except: continue
        print(f"[Dataset] Found {len(self.items)} cycles.")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        wav = load_wav_16k(it.wav_path)
        if wav is None: return None
        start, end = int(it.start_s * SR), int(it.end_s * SR)
        seg = wav[:, start:end]
        target_len = int(self.target_sec * SR)
        if seg.shape[-1] == 0: seg = torch.zeros(1, target_len)
        if seg.shape[-1] < target_len:
            seg = nn.functional.pad(seg, (0, target_len - seg.shape[-1]))
        elif seg.shape[-1] > target_len:
            off = np.random.randint(0, seg.shape[-1] - target_len + 1) if self.random_crop else (seg.shape[-1] - target_len) // 2
            seg = seg[:, off:off + target_len]
        if self.augment: seg = apply_random_augment(seg)
        S = self.amp2db(self.melspec(seg) + 1e-10)
        S = (S - S.mean()) / (S.std() + 1e-6)
        return S, it.y

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None, None
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
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class ScratchFeatureExtractor(nn.Module):
    def __init__(self, n_mels=40):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 32), nn.MaxPool2d(2),
            ConvBlock(32, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
            ConvBlock(64, 128), nn.Dropout(0.1))
        conv_dim = (n_mels // 4) * 128
        self.rnn = nn.LSTM(conv_dim, 256, num_layers=2, batch_first=True,
                           dropout=0.1, bidirectional=True)
    def forward(self, x):
        B, _, M, T = x.shape
        h = self.features(x)
        h = h.permute(0, 3, 1, 2).contiguous().view(B, h.shape[3], -1)
        y, _ = self.rnn(h)
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

# ------------------- SPLITTING & BALANCING ------------------- #
def oversample_indices(base_dataset, indices, seed=SEED, out_csv=None):
    """Balance classes by oversampling indices to the max class count."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    label_to_idx = defaultdict(list)
    for i in indices:
        y = base_dataset.items[int(i)].y
        label_to_idx[y].append(int(i))
    counts = {y: len(v) for y, v in label_to_idx.items()}
    max_count = max(counts.values()) if counts else 0
    expanded = []
    for y, idxs in label_to_idx.items():
        need = max_count - len(idxs)
        if need > 0:
            extra = rng.choice(idxs, size=need, replace=True).tolist()
            merged = idxs + extra
        else:
            merged = idxs
        expanded.extend(merged)
    rng.shuffle(expanded)
    if out_csv:
        pd.DataFrame([{"idx": i, "label": base_dataset.items[i].y,
                       "label_name": LABELS[base_dataset.items[i].y]}
                      for i in expanded]).to_csv(out_csv, index=False)
        print(f"[Oversample] Per-class counts={counts} -> expanded to {len(expanded)} total (per-class={max_count})")
    return np.array(expanded)

def groupwise_split_indices(base_dataset, test_size=0.15, val_size_within_remain=0.15, seed=SEED):
    """Group-wise split by subject IDs:
       1) Test groups via GroupShuffleSplit (approx 15% by samples)
       2) From the remaining groups, Train/Val via GroupShuffleSplit (~15% of remain to Val)
       Returns: train_idx, val_idx, test_idx (arrays of item indices)
    """
    items = base_dataset.items
    groups = np.array([subject_id_of(ci.wav_path) for ci in items])
    all_idx = np.arange(len(items))

    # Helper to compute sample coverage by selected groups
    def idx_from_groups(selected_groups):
        mask = np.isin(groups, selected_groups)
        return all_idx[mask]

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(all_idx, groups=groups))
    # This gives group-wise split; percentage by samples is approximate.

    # Now split train_val into train/val by groups
    groups_remain = groups[train_val_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_within_remain, random_state=seed)
    train_idx_rel, val_idx_rel = next(gss_val.split(train_val_idx, groups=groups_remain))

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    # Print actual proportions
    N = len(all_idx)
    print(f"[Group Split] N={N}  test={len(test_idx)} ({len(test_idx)/N:.3%})  "
          f"train={len(train_idx)} ({len(train_idx)/N:.3%})  val={len(val_idx)} ({len(val_idx)/N:.3%})")
    return train_idx, val_idx, test_idx

# ------------------- SAVE HELPERS ------------------- #
def save_hyperparams(outdir):
    HYPERPARAMS = {
        "SEED": SEED,
        "EPOCHS": EPOCHS,
        "BATCH_TRAIN": BATCH_TRAIN,
        "BATCH_VAL": BATCH_VAL,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "TARGET_SEC": TARGET_SEC,
        "BALANCE_DATA": BALANCE_DATA,
        "PATIENCE": PATIENCE,
        "SR": SR,
        "WIN": WIN,
        "HOP": HOP,
        "N_MELS": N_MELS,
        "FMIN": FMIN,
        "FMAX": FMAX,
        "LABELS": ",".join(LABELS)
    }
    df = pd.DataFrame({"hyperparameter": list(HYPERPARAMS.keys()),
                       "value": list(HYPERPARAMS.values())})
    df.to_csv(os.path.join(outdir, "hyperparameters.csv"), index=False)
    return df

def save_confusions(y_true, y_pred, labels, outdir, prefix):
    """Overall confusion + per-class (one-vs-rest) CSV + PNG."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(os.path.join(outdir, f"{prefix}_confusion_matrix.csv"))
    plt.figure(figsize=(6,5)); sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.title(f"Confusion Matrix ({prefix})")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{prefix}_confusion_matrix.png"), dpi=150); plt.close()
    # one-vs-rest
    for i, name in enumerate(labels):
        yt_bin = (y_true == i).astype(int)
        yp_bin = (y_pred == i).astype(int)
        cm_i = confusion_matrix(yt_bin, yp_bin, labels=[0,1])
        df_i = pd.DataFrame(cm_i,
                            index=[f"Actual not {name}", f"Actual {name}"],
                            columns=[f"Pred not {name}", f"Pred {name}"])
        df_i.to_csv(os.path.join(outdir, f"{prefix}_confusion_{name}_onevsrest.csv"))
        plt.figure(figsize=(4.2,3.6)); sns.heatmap(df_i, annot=True, fmt="d", cmap="Purples")
        plt.title(f"{name}: One-vs-Rest ({prefix})"); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{prefix}_confusion_{name}_onevsrest.png"), dpi=150); plt.close()

def save_multiclass_roc(y_true, y_prob, labels, outdir, prefix):
    """ROC curves + AUC (per-class, micro, macro)."""
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    if y_prob.ndim != 2 or y_prob.shape[1] != len(labels):
        print("[Warn] ROC skipped due to empty or invalid prob array.")
        return
    Y_true = label_binarize(y_true, classes=list(range(len(labels))))
    fpr, tpr, roc_auc = {}, {}, {}
    for i, _ in enumerate(labels):
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
    # save table
    rows = [{"class": labels[i], "AUC": roc_auc[i]} for i in range(len(labels))]
    rows += [{"class": "micro", "AUC": roc_auc["micro"]}, {"class": "macro", "AUC": roc_auc["macro"]}]
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{prefix}_auc.csv"), index=False)
    # plot
    plt.figure(figsize=(7,6))
    for i, name in enumerate(labels):
        plt.plot(fpr[i], tpr[i], lw=1.6, label=f"{name} (AUC={roc_auc[i]:.3f})")
    plt.plot(fpr["micro"], tpr["micro"], lw=2.0, linestyle="--", label=f"micro (AUC={roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], lw=2.0, linestyle="-.", label=f"macro (AUC={roc_auc['macro']:.3f})")
    plt.plot([0,1],[0,1],"k:", lw=1)
    plt.xlim([0,1]); plt.ylim([0,1.05]); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({prefix})"); plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{prefix}_roc.png"), dpi=160); plt.close()

# ------------------- MAIN ------------------- #
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Main] Using device:", device)

    # Save hyperparameters table
    _ = save_hyperparams(OUTDIR)

    # Base dataset (no aug) to enumerate items/labels/groups
    base = ICBHICycleDataset(DATASET_ROOT, TARGET_SEC, random_crop=True, augment=False)
    all_idx = np.arange(len(base))
    all_labels = np.array([it.y for it in base.items])
    groups = np.array([subject_id_of(it.wav_path) for it in base.items])

    # ---- Group-wise TEST split (~15% by samples; no leakage) ----
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val_idx, test_idx = next(gss_test.split(all_idx, groups=groups))
    print(f"[Group Test Split] test={len(test_idx)} ({len(test_idx)/len(all_idx):.3%})")

    # ---- Group-wise Train/Val split from remaining (~85/15 of remain) ----
    groups_remain = groups[train_val_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_idx_rel, val_idx_rel = next(gss_val.split(train_val_idx, groups=groups_remain))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    print(f"[Group Split Totals] N={len(all_idx)}  "
          f"train={len(train_idx)} ({len(train_idx)/len(all_idx):.3%})  "
          f"val={len(val_idx)} ({len(val_idx)/len(all_idx):.3%})  "
          f"test={len(test_idx)} ({len(test_idx)/len(all_idx):.3%})")

    # ---- Oversample TRAIN ONLY (balance) ----
    if BALANCE_DATA:
        train_idx_bal = oversample_indices(base, train_idx,
                                           seed=SEED,
                                           out_csv=os.path.join(OUTDIR, "train_oversampled.csv"))
    else:
        train_idx_bal = train_idx

    # ---- Datasets / Dataloaders (train uses augmentation) ----
    train_ds = Subset(ICBHICycleDataset(DATASET_ROOT, TARGET_SEC, True,  augment=True),  train_idx_bal)
    val_ds   = Subset(ICBHICycleDataset(DATASET_ROOT, TARGET_SEC, False, augment=False), val_idx.tolist())
    test_ds  = Subset(ICBHICycleDataset(DATASET_ROOT, TARGET_SEC, False, augment=False), test_idx.tolist())

    train_dl = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,  collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_VAL,   shuffle=False, collate_fn=collate_fn)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_VAL,   shuffle=False, collate_fn=collate_fn)

    # ---- Model / Optimizer / Loss ----
    model = ICBHIClassifier(N_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    history, best_mf1, no_improve = [], -1, 0

    # ---- Train loop ----
    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        model.train()
        yt, yp, losses = [], [], []

        for X, y in tqdm(train_dl, desc=f"Training epoch {epoch}", leave=False):
            if X is None: continue
            X, y = X.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(X)
                loss = crit(out, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            yp.extend(out.argmax(1).detach().cpu().numpy())
            yt.extend(y.detach().cpu().numpy())

        tr_acc = accuracy_score(yt, yp)
        tr_mf1 = f1_score(yt, yp, average="macro")

        # ---- Validation ----
        model.eval()
        yt_val, yp_val, vlosses, yprob_val = [], [], [], []
        with torch.no_grad():
            for X, y in tqdm(val_dl, desc=f"Validation epoch {epoch}", leave=False):
                if X is None: continue
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = crit(out, y)
                vlosses.append(loss.item())
                probs = torch.softmax(out, dim=1).cpu().numpy()
                yprob_val.append(probs)
                yp_val.extend(out.argmax(1).cpu().numpy())
                yt_val.extend(y.cpu().numpy())

        yprob_val = np.vstack(yprob_val) if len(yprob_val) > 0 else np.zeros((0, N_CLASSES))

        # ---- Metrics & report ----
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
            "val_f1_macro": f1_score(yt_val, yp_val, average="macro", zero_division=0)
        }
        history.append(metrics)

        report = classification_report(yt_val, yp_val, target_names=LABELS, output_dict=True, zero_division=0)
        df_rep = pd.DataFrame(report).T
        df_rep["epoch"] = epoch
        df_rep.to_csv(os.path.join(OUTDIR, f"metrics_per_class_epoch{epoch:03d}.csv"))

        print(f"[Epoch {epoch}] Train Acc={tr_acc:.3f} Val Acc={va_acc:.3f} Val F1={va_mf1:.3f}")

        # ---- Early stopping ----
        if va_mf1 > best_mf1:
            best_mf1, no_improve = va_mf1, 0
            torch.save(model.state_dict(), os.path.join(OUTDIR, "best.ckpt"))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs → early stop.")
                break

    # ---- Save learning curves ----
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(OUTDIR, "metrics_per_epoch.csv"), index=False)

    plt.figure(figsize=(7,5))
    if len(hist_df) > 0:
        plt.plot(hist_df["epoch"], hist_df["train_mf1"], label="Train F1", marker="o")
        plt.plot(hist_df["epoch"], hist_df["val_mf1"], label="Val F1", marker="o")
        plt.plot(hist_df["epoch"], hist_df["val_acc"], label="Val Acc", marker="x")
    plt.xlabel("Epoch"); plt.ylabel("Score")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "learning_curves.png"), dpi=150)
    plt.close()

    # ---- Evaluate BEST checkpoint on VAL and TEST (CM + ROC/AUC) ----
    best_path = os.path.join(OUTDIR, "best.ckpt")
    if os.path.isfile(best_path):
        best_model = ICBHIClassifier(N_CLASSES).to(device)
        best_model.load_state_dict(torch.load(best_path, map_location=device))
        best_model.eval()

        # VAL
        yt_best, yp_best, yprob_best = [], [], []
        with torch.no_grad():
            for X, y in tqdm(val_dl, desc="Best model eval (val set)"):
                if X is None: continue
                X = X.to(device)
                out = best_model(X)
                yp_best.extend(out.argmax(1).cpu().numpy())
                yprob_best.append(torch.softmax(out, dim=1).cpu().numpy())
                yt_best.extend(y.numpy())
        yprob_best = np.vstack(yprob_best) if len(yprob_best) else np.zeros((0, N_CLASSES))
        save_confusions(yt_best, yp_best, LABELS, OUTDIR, prefix="best_val")
        if len(yprob_best):
            save_multiclass_roc(yt_best, yprob_best, LABELS, OUTDIR, prefix="best_val")

        # TEST
        yt_test, yp_test, yprob_test = [], [], []
        with torch.no_grad():
            for X, y in tqdm(test_dl, desc="Best model eval (test set)"):
                if X is None: continue
                X = X.to(device)
                out = best_model(X)
                yp_test.extend(out.argmax(1).cpu().numpy())
                yprob_test.append(torch.softmax(out, dim=1).cpu().numpy())
                yt_test.extend(y.numpy())
        yprob_test = np.vstack(yprob_test) if len(yprob_test) else np.zeros((0, N_CLASSES))
        save_confusions(yt_test, yp_test, LABELS, OUTDIR, prefix="best_test")
        if len(yprob_test):
            save_multiclass_roc(yt_test, yprob_test, LABELS, OUTDIR, prefix="best_test")

if __name__ == "__main__":
    main()
