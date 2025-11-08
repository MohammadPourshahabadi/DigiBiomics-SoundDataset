# -*- coding: utf-8 -*-
"""
ICBHI 2017 cycle classification (from scratch, no pretrain) with:
- Strict patient-wise splits (no leakage) using subject ID from filenames.
- Balanced group-wise split: train/val/test label distributions closer to global.
- Class-weighted CrossEntropyLoss instead of hard oversampling.
- Light waveform augmentations on TRAIN only (gain, noise, shift).
- Early stopping on val macro-F1.
- Mixed precision (if GPU available).
- Epoch metrics + per-class reports.
- Confusion matrices (overall + one-vs-rest).
- ROC + AUC (per-class, micro, macro) for Val/Test.
- Learning curves and hyperparameters table.
- Test metrics CSVs (accuracy, precision/recall/F1 micro&macro,
  balanced acc, kappa, MCC, per-class report).
"""

import os, glob, random, math
from dataclasses import dataclass
from collections import Counter, defaultdict

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

OUTDIR = "/content/drive/MyDrive/Colab Notebooks/Datasets/Results_ICBHI_Scratch_FullMetrics_balanced_groups_classweights"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------- HYPERPARAMS ------------------- #
SEED = 1337
EPOCHS = 10
BATCH_TRAIN = 16
BATCH_VAL = 32
LR = 1.5e-4
WEIGHT_DECAY = 1e-3
TARGET_SEC = 2.5
PATIENCE = 8                   # early stopping patience (stricter)
USE_CLASS_WEIGHTS = True       # use class-weighted loss (no hard oversampling)

SR = 16000
WIN = int(0.025 * SR)          # 25 ms
HOP = int(0.010 * SR)          # 10 ms
N_MELS = 40
FMIN = 20
FMAX = 8000

LABELS = ["normal", "crackle", "wheeze", "both"]
N_CLASSES = len(LABELS)

# Optional quick peek
print("Sample .wav files found:")
for f in glob.glob(os.path.join(BASE_PATH, "**/*.wav"), recursive=True)[:5]:
    print("  ", f)
print("Sample .txt files found:")
for f in glob.glob(os.path.join(BASE_PATH, "**/*.txt"), recursive=True)[:5]:
    print("  ", f)

# ------------------- UTILS ------------------- #
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cycle_label(c, w):
    if c == 0 and w == 0: return 0          # normal
    if c == 1 and w == 0: return 1          # crackle
    if c == 0 and w == 1: return 2          # wheeze
    return 3                                # both

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

# ------------------- AUGMENTATIONS (milder) ------------------- #
def rand_bool(p): 
    return random.random() < p

def add_gaussian_noise(wav, snr_db_range=(18, 30)):
    x = wav.squeeze(0)
    rms = torch.sqrt(torch.mean(x ** 2) + 1e-12)
    snr_db = random.uniform(*snr_db_range)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_rms = rms / math.sqrt(snr_lin)
    noise = torch.randn_like(x) * noise_rms
    return (x + noise).unsqueeze(0)

def random_gain(wav, db_range=(-3, 3)):
    gain_db = random.uniform(*db_range)
    gain = 10.0 ** (gain_db / 20.0)
    return wav * gain

def time_shift(wav, max_shift_sec=0.1, sr=16000):
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
    if rand_bool(0.3):
        wav = random_gain(wav, (-3, 3))
    if rand_bool(0.3):
        wav = add_gaussian_noise(wav, (18, 30))
    if rand_bool(0.3):
        wav = time_shift(wav, 0.1, SR)
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
            f_min=FMIN, f_max=FMAX, n_mels=N_MELS, center=True, power=2.0
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
                            self.items.append(CycleItem(
                                cand_wav, s, e, cycle_label(c, w)
                            ))
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
            conv_dim, 256, num_layers=2, batch_first=True,
            dropout=0.1, bidirectional=True
        )

    def forward(self, x):
        B, _, M, T = x.shape
        h = self.features(x)                # (B, C, M', T')
        # treat time dimension as sequence
        h = h.permute(0, 3, 1, 2).contiguous()
        h = h.view(B, h.shape[1], -1)       # (B, T', C*M')
        y, _ = self.rnn(h)
        return y                            # (B, T', 512)

class ICBHIClassifier(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.enc = ScratchFeatureExtractor(N_MELS)
        self.attn = nn.Linear(512, 1)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        seq = self.enc(x)                   # (B, T', 512)
        w = torch.softmax(self.attn(seq).squeeze(-1), dim=1)  # (B, T')
        emb = (seq * w.unsqueeze(-1)).sum(1)                  # (B, 512)
        return self.head(emb)

# ------------------- SPLITTING: BALANCED GROUP SPLITS ------------------- #
def groupwise_balanced_splits(base_dataset,
                              test_size=0.15,
                              val_size_within_remain=0.15,
                              max_trials=200,
                              seed=SEED):
    """
    Patient-wise (group) splits.
    We sample many group-wise splits and pick the one where
    train/val/test label distributions are closest to the global distribution.
    """
    all_idx = np.arange(len(base_dataset))
    groups = np.array([subject_id_of(it.wav_path) for it in base_dataset.items])
    y = np.array([it.y for it in base_dataset.items])

    rng = np.random.RandomState(seed)
    best_score = float("inf")
    best = None

    def label_dist(idxs):
        c = np.bincount(y[idxs], minlength=N_CLASSES).astype(float)
        s = c.sum()
        return c / s if s > 0 else np.zeros_like(c)

    p_all = label_dist(all_idx)

    for _ in range(max_trials):
        rs = int(rng.randint(0, 1_000_000))

        # 1) test split (group-wise)
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=rs
        )
        train_val_idx, test_idx = next(
            gss_test.split(all_idx, groups=groups)
        )

        # 2) val split from remaining (group-wise)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_size_within_remain, random_state=rs + 1
        )
        train_idx_rel, val_idx_rel = next(
            gss_val.split(train_val_idx, groups=groups[train_val_idx])
        )
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]

        # distributions
        p_tr = label_dist(train_idx)
        p_va = label_dist(val_idx)
        p_te = label_dist(test_idx)

        # objective: how far each is from global
        score = ((p_tr - p_all) ** 2).sum() \
              + ((p_va - p_all) ** 2).sum() \
              + ((p_te - p_all) ** 2).sum()

        if score < best_score:
            best_score = score
            best = (train_idx, val_idx, test_idx)

    train_idx, val_idx, test_idx = best

    print("[Balanced Group Split Selected]")
    print(" Global:", Counter(y))
    print(" Train :", len(train_idx), Counter(y[train_idx]))
    print(" Val   :", len(val_idx),   Counter(y[val_idx]))
    print(" Test  :", len(test_idx),  Counter(y[test_idx]))

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
        "PATIENCE": PATIENCE,
        "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,
        "SR": SR,
        "WIN": WIN,
        "HOP": HOP,
        "N_MELS": N_MELS,
        "FMIN": FMIN,
        "FMAX": FMAX,
        "LABELS": ",".join(LABELS),
    }
    df = pd.DataFrame({
        "hyperparameter": list(HYPERPARAMS.keys()),
        "value": list(HYPERPARAMS.values())
    })
    df.to_csv(os.path.join(outdir, "hyperparameters.csv"), index=False)
    return df

def save_confusions(y_true, y_pred, labels, outdir, prefix):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(os.path.join(outdir, f"{prefix}_confusion_matrix.csv"))

    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix ({prefix})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_confusion_matrix.png"), dpi=150)
    plt.close()

    # one-vs-rest
    for i, name in enumerate(labels):
        yt_bin = (y_true == i).astype(int)
        yp_bin = (y_pred == i).astype(int)
        cm_i = confusion_matrix(yt_bin, yp_bin, labels=[0, 1])
        df_i = pd.DataFrame(
            cm_i,
            index=[f"Actual not {name}", f"Actual {name}"],
            columns=[f"Pred not {name}", f"Pred {name}"],
        )
        df_i.to_csv(os.path.join(outdir, f"{prefix}_confusion_{name}_onevsrest.csv"))

        plt.figure(figsize=(4.2, 3.6))
        sns.heatmap(df_i, annot=True, fmt="d", cmap="Purples")
        plt.title(f"{name}: One-vs-Rest ({prefix})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f"{prefix}_confusion_{name}_onevsrest.png"),
            dpi=150,
        )
        plt.close()

def save_multiclass_roc(y_true, y_prob, labels, outdir, prefix):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim != 2 or y_prob.shape[1] != len(labels) or len(y_prob) == 0:
        print("[Warn] ROC skipped due to empty or invalid prob array.")
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
    all_fpr = np.unique(
        np.concatenate([fpr[i] for i in range(len(labels))])
    )
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(labels)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(labels)

    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # table
    rows = [{"class": labels[i], "AUC": roc_auc[i]} for i in range(len(labels))]
    rows += [
        {"class": "micro", "AUC": roc_auc["micro"]},
        {"class": "macro", "AUC": roc_auc["macro"]},
    ]
    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, f"{prefix}_auc.csv"), index=False
    )

    # plot
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(labels):
        plt.plot(fpr[i], tpr[i], lw=1.6, label=f"{name} (AUC={roc_auc[i]:.3f})")
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        lw=2.0,
        linestyle="--",
        label=f"micro (AUC={roc_auc['micro']:.3f})",
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        lw=2.0,
        linestyle="-.",
        label=f"macro (AUC={roc_auc['macro']:.3f})",
    )
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
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

    # Save hyperparameters table
    _ = save_hyperparams(OUTDIR)

    # Base dataset (no aug) only to define items/labels/groups for splitting
    base = ICBHICycleDataset(DATASET_ROOT, TARGET_SEC, random_crop=True, augment=False)

    # Group-wise, distribution-aware splits
    train_idx, val_idx, test_idx = groupwise_balanced_splits(
        base,
        test_size=0.15,
        val_size_within_remain=0.15,
        max_trials=200,
        seed=SEED,
    )

    # ----- Class weights from TRAIN (no leakage) -----
    if USE_CLASS_WEIGHTS:
        y_train = [base.items[i].y for i in train_idx]
        counts = Counter(y_train)
        total = sum(counts.values())
        # inverse-frequency style
        class_weights = [
            total / (N_CLASSES * counts[c]) if counts[c] > 0 else 0.0
            for c in range(N_CLASSES)
        ]
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print("[Class Weights]", class_weights)
    else:
        class_weights_tensor = None

    # ----- Datasets / Loaders -----
    # Train with augmentation
    train_ds_full = ICBHICycleDataset(
        DATASET_ROOT, TARGET_SEC, random_crop=True, augment=True
    )
    val_ds_full = ICBHICycleDataset(
        DATASET_ROOT, TARGET_SEC, random_crop=False, augment=False
    )
    test_ds_full = ICBHICycleDataset(
        DATASET_ROOT, TARGET_SEC, random_crop=False, augment=False
    )

    train_ds = Subset(train_ds_full, train_idx.tolist())
    val_ds = Subset(val_ds_full, val_idx.tolist())
    test_ds = Subset(test_ds_full, test_idx.tolist())

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_TRAIN, shuffle=True, collate_fn=collate_fn
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_VAL, shuffle=False, collate_fn=collate_fn
    )
    test_dl = DataLoader(
        test_ds, batch_size=BATCH_VAL, shuffle=False, collate_fn=collate_fn
    )

    # ----- Model / Optimizer / Loss -----
    model = ICBHIClassifier(N_CLASSES).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights_tensor)
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    history = []
    best_mf1 = -1.0
    no_improve = 0

    # ------------------- TRAINING LOOP ------------------- #
    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        # ---- Train ----
        model.train()
        yt_tr, yp_tr, losses_tr = [], [], []

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

            losses_tr.append(loss.item())
            yp_tr.extend(out.argmax(1).detach().cpu().numpy())
            yt_tr.extend(y.detach().cpu().numpy())

        tr_acc = accuracy_score(yt_tr, yp_tr)
        tr_mf1 = f1_score(yt_tr, yp_tr, average="macro")

        # ---- Validation ----
        model.eval()
        yt_va, yp_va, losses_va, yprob_va = [], [], [], []

        with torch.no_grad():
            for X, y in tqdm(val_dl, desc=f"Validation epoch {epoch}", leave=False):
                if X is None:
                    continue
                X, y = X.to(device), y.to(device)

                out = model(X)
                loss = crit(out, y)

                losses_va.append(loss.item())
                probs = torch.softmax(out, dim=1).cpu().numpy()
                yprob_va.append(probs)
                yp_va.extend(out.argmax(1).cpu().numpy())
                yt_va.extend(y.cpu().numpy())

        if len(yprob_va) > 0:
            yprob_va = np.vstack(yprob_va)
        else:
            yprob_va = np.zeros((0, N_CLASSES))

        va_acc = accuracy_score(yt_va, yp_va)
        va_mf1 = f1_score(yt_va, yp_va, average="macro")

        metrics = {
            "epoch": epoch,
            "train_acc": tr_acc,
            "val_acc": va_acc,
            "train_mf1": tr_mf1,
            "val_mf1": va_mf1,
            "val_precision_micro": precision_score(yt_va, yp_va, average="micro", zero_division=0),
            "val_recall_micro": recall_score(yt_va, yp_va, average="micro", zero_division=0),
            "val_f1_micro": f1_score(yt_va, yp_va, average="micro", zero_division=0),
            "val_precision_macro": precision_score(yt_va, yp_va, average="macro", zero_division=0),
            "val_recall_macro": recall_score(yt_va, yp_va, average="macro", zero_division=0),
            "val_f1_macro": va_mf1,
            "train_loss": np.mean(losses_tr) if losses_tr else 0.0,
            "val_loss": np.mean(losses_va) if losses_va else 0.0,
        }
        history.append(metrics)

        # per-epoch per-class report (val)
        rep = classification_report(
            yt_va, yp_va, target_names=LABELS,
            output_dict=True, zero_division=0
        )
        df_rep = pd.DataFrame(rep).T
        df_rep["epoch"] = epoch
        df_rep.to_csv(
            os.path.join(OUTDIR, f"metrics_per_class_epoch{epoch:03d}.csv")
        )

        print(
            f"[Epoch {epoch}] "
            f"Train Acc={tr_acc:.3f}  Train F1={tr_mf1:.3f} | "
            f"Val Acc={va_acc:.3f}  Val F1={va_mf1:.3f}"
        )

        # ---- Early stopping on val macro-F1 ----
        if va_mf1 > best_mf1:
            best_mf1 = va_mf1
            no_improve = 0
            torch.save(
                model.state_dict(),
                os.path.join(OUTDIR, "best.ckpt")
            )
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(
                    f"No val F1 improvement for {PATIENCE} epochs → early stop."
                )
                break

    # ---- Save learning curves ----
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(
        os.path.join(OUTDIR, "metrics_per_epoch.csv"), index=False
    )

    plt.figure(figsize=(7, 5))
    if len(hist_df) > 0:
        plt.plot(
            hist_df["epoch"],
            hist_df["train_mf1"],
            label="Train F1",
            marker="o",
        )
        plt.plot(
            hist_df["epoch"],
            hist_df["val_mf1"],
            label="Val F1",
            marker="o",
        )
        plt.plot(
            hist_df["epoch"],
            hist_df["val_acc"],
            label="Val Acc",
            marker="x",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTDIR, "learning_curves.png"), dpi=150
    )
    plt.close()

    # ------------------- EVAL BEST ON VAL + TEST ------------------- #
    best_path = os.path.join(OUTDIR, "best.ckpt")
    if os.path.isfile(best_path):
        best_model = ICBHIClassifier(N_CLASSES).to(device)
        best_model.load_state_dict(
            torch.load(best_path, map_location=device)
        )
        best_model.eval()

        # ----- VAL -----
        yt_v, yp_v, yprob_v = [], [], []
        with torch.no_grad():
            for X, y in tqdm(val_dl, desc="Best model eval (val set)"):
                if X is None:
                    continue
                X = X.to(device)
                out = best_model(X)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                yprob_v.append(probs)
                yp_v.extend(out.argmax(1).cpu().numpy())
                yt_v.extend(y.numpy())
        if len(yprob_v) > 0:
            yprob_v = np.vstack(yprob_v)
        else:
            yprob_v = np.zeros((0, N_CLASSES))

        save_confusions(yt_v, yp_v, LABELS, OUTDIR, prefix="best_val")
        if len(yprob_v):
            save_multiclass_roc(
                yt_v, yprob_v, LABELS, OUTDIR, prefix="best_val"
            )

        # ----- TEST -----
        yt_t, yp_t, yprob_t = [], [], []
        with torch.no_grad():
            for X, y in tqdm(test_dl, desc="Best model eval (test set)"):
                if X is None:
                    continue
                X = X.to(device)
                out = best_model(X)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                yprob_t.append(probs)
                yp_t.extend(out.argmax(1).cpu().numpy())
                yt_t.extend(y.numpy())
        if len(yprob_t) > 0:
            yprob_t = np.vstack(yprob_t)
        else:
            yprob_t = np.zeros((0, N_CLASSES))

        save_confusions(yt_t, yp_t, LABELS, OUTDIR, prefix="best_test")
        if len(yprob_t):
            save_multiclass_roc(
                yt_t, yprob_t, LABELS, OUTDIR, prefix="best_test"
            )

        # ---- Test metrics ----
        test_metrics = {
            "test_accuracy": accuracy_score(yt_t, yp_t),
            "test_precision_micro": precision_score(
                yt_t, yp_t, average="micro", zero_division=0
            ),
            "test_recall_micro": recall_score(
                yt_t, yp_t, average="micro", zero_division=0
            ),
            "test_f1_micro": f1_score(
                yt_t, yp_t, average="micro", zero_division=0
            ),
            "test_precision_macro": precision_score(
                yt_t, yp_t, average="macro", zero_division=0
            ),
            "test_recall_macro": recall_score(
                yt_t, yp_t, average="macro", zero_division=0
            ),
            "test_f1_macro": f1_score(
                yt_t, yp_t, average="macro", zero_division=0
            ),
            "test_balanced_accuracy": balanced_accuracy_score(
                yt_t, yp_t
            ),
            "test_kappa": cohen_kappa_score(yt_t, yp_t),
            "test_mcc": matthews_corrcoef(yt_t, yp_t),
        }
        pd.DataFrame([test_metrics]).to_csv(
            os.path.join(OUTDIR, "best_test_metrics_summary.csv"),
            index=False,
        )

        test_report = classification_report(
            yt_t, yp_t, target_names=LABELS,
            output_dict=True, zero_division=0
        )
        pd.DataFrame(test_report).T.to_csv(
            os.path.join(OUTDIR, "best_test_classification_report.csv")
        )

        print(
            "[Best/Test] Acc={acc:.3f}  F1(macro)={f1m:.3f}  "
            "F1(micro)={f1mi:.3f}  BalAcc={ba:.3f}  κ={kap:.3f}  MCC={mcc:.3f}"
            .format(
                acc=test_metrics["test_accuracy"],
                f1m=test_metrics["test_f1_macro"],
                f1mi=test_metrics["test_f1_micro"],
                ba=test_metrics["test_balanced_accuracy"],
                kap=test_metrics["test_kappa"],
                mcc=test_metrics["test_mcc"],
            )
        )

if __name__ == "__main__":
    main()
