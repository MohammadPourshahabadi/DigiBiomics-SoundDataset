# -*- coding: utf-8 -*-
"""
Slightly deeper CRNN baseline for ICBHI cycles:

- Load all cycles from .txt annotations
- Patient-wise split: 70% train, 15% val, 15% test (by subject ID from filename)
- Print stats (samples, patients, class counts) for each split
- Convert to log-mel spectrograms
- Train a deeper CNN + 2-layer BiLSTM model from scratch
- Report:
    * number of trainable parameters
    * train/val/test Accuracy per epoch
    * train/val/test micro-F1 per epoch
- Save:
    * train/val/test stats CSVs
    * metrics_per_epoch.csv
    * model_info.txt
    * accuracy_curves.png
    * micro_f1_curves.png
    * confusion_matrix_test.png + confusion_matrix_test.csv
    * roc_test.png + roc_test_auc.csv
"""

import os
import glob
import random
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import pandas as pd

import soundfile as sf
import librosa

# ---------------------------------------------------------------------#
# PATHS & CONFIG
# ---------------------------------------------------------------------#
BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/Datasets/ICBHI 2017 Challenge/ICBHI_final_database"

RESULT_ROOT = "/content/drive/MyDrive/Colab Notebooks/Datasets"
RESULT_DIR = os.path.join(RESULT_ROOT, "Resultfornewmodel")
os.makedirs(RESULT_DIR, exist_ok=True)
print("Saving results to:", RESULT_DIR)

SR = 16000
TARGET_SEC = 2.5
N_MELS = 40
BATCH_TRAIN = 16
BATCH_VAL = 32
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 1337

LABELS = ["normal", "crackle", "wheeze", "both"]
N_CLASSES = len(LABELS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------------------------------------------------#
# UTILS
# ---------------------------------------------------------------------#
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subject_id_from_path(path: str) -> str:
    """Extract subject ID from filename like '221_2b1_...wav' -> '221'."""
    base = os.path.basename(path)
    return base.split("_")[0]


def cycle_label(c, w):
    """Map crackle/wheeze flags to 4 classes."""
    if c == 0 and w == 0:
        return 0  # normal
    if c == 1 and w == 0:
        return 1  # crackle
    if c == 0 and w == 1:
        return 2  # wheeze
    return 3      # both


def load_wav_safe(path, target_sr=16000):
    """Load audio as mono float32 tensor and resample to target_sr."""
    try:
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
    except Exception:
        data, sr = librosa.load(path, sr=None, mono=True)

    wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, N)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav, target_sr


# ---------------------------------------------------------------------#
# DATASET: extract cycles from txt + wav, convert to log-mel
# ---------------------------------------------------------------------#
@dataclass
class CycleItem:
    wav_path: str
    pid: str
    start_s: float
    end_s: float
    label: int


class ICBHICycleDataset(Dataset):
    def __init__(self, root, target_sec=2.5, sr=16000):
        self.root = root
        self.sr = sr
        self.target_len = int(target_sec * sr)
        self.items = []
        self._index_cycles()

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            win_length=int(0.025 * sr),
            hop_length=int(0.010 * sr),
            f_min=20,
            f_max=8000,
            n_mels=N_MELS,
            power=2.0,
            center=True,
        )
        self.amp2db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def _index_cycles(self):
        txts = sorted(glob.glob(os.path.join(self.root, "**/*.txt"), recursive=True))
        wavs = sorted(glob.glob(os.path.join(self.root, "**/*.wav"), recursive=True))
        wav_map = {os.path.basename(w).lower(): w for w in wavs}

        for txt in txts:
            cand = txt[:-4] + ".wav"
            if not os.path.isfile(cand):
                cand = wav_map.get(os.path.basename(cand).lower())
                if cand is None:
                    continue
            pid = subject_id_from_path(cand)
            with open(txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    try:
                        s, e = float(parts[0]), float(parts[1])
                        c, w = int(float(parts[2])), int(float(parts[3]))
                    except ValueError:
                        continue
                    if e <= s:
                        continue
                    y = cycle_label(c, w)
                    self.items.append(CycleItem(cand, pid, s, e, y))
        print(f"Indexed {len(self.items)} cycles from {len(txts)} txt files.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        wav, sr = load_wav_safe(item.wav_path, self.sr)

        start = int(item.start_s * self.sr)
        end = int(item.end_s * self.sr)
        seg = wav[:, start:end]

        if seg.shape[-1] <= 0:
            seg = torch.zeros(1, self.target_len)
        elif seg.shape[-1] < self.target_len:
            seg = nn.functional.pad(seg, (0, self.target_len - seg.shape[-1]))
        else:
            # centre crop to TARGET_SEC
            off = (seg.shape[-1] - self.target_len) // 2
            seg = seg[:, off:off + self.target_len]

        S = self.melspec(seg)           # (1, n_mels, T)
        S_db = self.amp2db(S + 1e-10)
        # per-sample standardisation
        S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
        return S_db, item.label, item.pid


def collate_fn(batch):
    X = torch.stack([b[0] for b in batch])
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    pids = [b[2] for b in batch]
    return X, y, pids


# ---------------------------------------------------------------------#
# PATIENT-WISE SPLIT 70 / 15 / 15
# ---------------------------------------------------------------------#
def patientwise_split(dataset, train_ratio=0.7, val_ratio=0.15, seed=SEED):
    groups = np.array([it.pid for it in dataset.items])
    idx = np.arange(len(dataset.items))

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, groups=groups))

    remaining_ratio = 1.0 - train_ratio
    val_rel = val_ratio / remaining_ratio
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_rel, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_idx, groups=groups[temp_idx]))

    print(f"Total cycles: {len(idx)}")
    print(f"Train: {len(train_idx)} ({len(train_idx)/len(idx):.1%})")
    print(f"Val:   {len(val_idx)} ({len(val_idx)/len(idx):.1%})")
    print(f"Test:  {len(test_idx)} ({len(test_idx)/len(idx):.1%})")
    return train_idx, val_idx, test_idx


def subset_stats(dataset, indices, name):
    sub_items = [dataset.items[i] for i in indices]
    pids = [it.pid for it in sub_items]
    labels = [it.label for it in sub_items]
    unique_pids = sorted(set(pids))
    counts = np.bincount(labels, minlength=N_CLASSES)

    print(f"\n[{name} stats]")
    print(f"  #cycles:   {len(indices)}")
    print(f"  #patients: {len(unique_pids)}")
    for k, lab in enumerate(LABELS):
        print(f"  {lab:7s}: {counts[k]}")

    # Save stats also to CSV
    df = pd.DataFrame({
        "split": [name],
        "n_cycles": [len(indices)],
        "n_patients": [len(unique_pids)],
        **{f"n_{LABELS[k]}": [counts[k]] for k in range(N_CLASSES)},
    })
    stats_path = os.path.join(RESULT_DIR, f"{name.lower()}_stats.csv")
    df.to_csv(stats_path, index=False)


# ---------------------------------------------------------------------#
# DEEPER CNN + 2-LAYER LSTM MODEL
# ---------------------------------------------------------------------#
class ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + ReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SimpleCRNN(nn.Module):
    """
    Slightly more complex CRNN:
    - 3 ConvBlocks (1->32->64->128) with 2 MaxPool layers + Dropout2d
    - 2-layer BiLSTM with hidden_size=128 and dropout
    - Dropout before final classifier
    """
    def __init__(self, n_mels=N_MELS, n_classes=N_CLASSES):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),            # downsample /2 (freq & time)
            ConvBlock(64, 128),
            nn.MaxPool2d(2),            # downsample /4 total
            nn.Dropout2d(0.2),
        )

        # after two MaxPool2d(2): n_mels -> n_mels / 4
        conv_dim = (n_mels // 4) * 128  # features per time step for LSTM

        self.lstm = nn.LSTM(
            input_size=conv_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 2, n_classes)
        )

    def forward(self, x):
        # x: (B,1,n_mels,T)
        h = self.conv(x)   # (B, C=128, M', T')
        B, C, M, T = h.shape
        # make time the sequence dimension: (B, T, C*M)
        h = h.permute(0, 3, 1, 2).contiguous().view(B, T, C * M)
        y, _ = self.lstm(h)         # (B, T, 2*hidden)
        emb = y.mean(dim=1)         # simple mean pooling over time -> (B, 256)
        logits = self.fc(emb)       # (B, n_classes)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------#
# TRAIN / EVAL HELPERS
# ---------------------------------------------------------------------#
def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    all_y, all_pred, all_prob = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for X, y, _ in loader:
        X = X.to(device)
        y = y.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(X)
            loss = criterion(logits, y)

            if train_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * y.size(0)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)

        all_y.extend(y.cpu().numpy())
        all_pred.extend(preds)
        all_prob.append(probs)

    all_prob = np.vstack(all_prob)
    avg_loss = total_loss / len(all_y)
    acc = accuracy_score(all_y, all_pred)
    micro_f1 = f1_score(all_y, all_pred, average="micro")

    return avg_loss, acc, micro_f1, np.array(all_y), np.array(all_pred), all_prob


def plot_curves(history, title, ylabel, savepath=None):
    epochs = np.arange(1, len(history["train"]) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train"], "-o", label="Train")
    plt.plot(epochs, history["val"], "-o", label="Val")
    plt.plot(epochs, history["test"], "-o", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150)
    plt.show()


def plot_confusion(y_true, y_pred, labels, title="Confusion matrix", savepath=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150)
        base, _ = os.path.splitext(savepath)
        np.savetxt(base + ".csv", cm, fmt="%d", delimiter=",")
    plt.show()


def plot_multiclass_roc(y_true, y_prob, labels, title="ROC", savepath=None):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_classes = len(labels)

    Y = label_binarize(y_true, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro
    fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(7, 6))
    for i, name in enumerate(labels):
        plt.plot(fpr[i], tpr[i], lw=1.5, label=f"{name} (AUC={roc_auc[i]:.3f})")
    plt.plot(fpr["micro"], tpr["micro"], "k--", lw=2, label=f"micro (AUC={roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], "k-.", lw=2, label=f"macro (AUC={roc_auc['macro']:.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150)
        # save AUC table
        rows = [{"class": labels[i], "AUC": roc_auc[i]} for i in range(n_classes)]
        rows += [{"class": "micro", "AUC": roc_auc["micro"]},
                 {"class": "macro", "AUC": roc_auc["macro"]}]
        base, _ = os.path.splitext(savepath)
        pd.DataFrame(rows).to_csv(base + "_auc.csv", index=False)
    plt.show()


# ---------------------------------------------------------------------#
# MAIN
# ---------------------------------------------------------------------#
def main():
    set_seed(SEED)

    # 1) Load base dataset
    base_ds = ICBHICycleDataset(BASE_PATH, TARGET_SEC, SR)

    # 2) Patient-wise split 70 / 15 / 15
    train_idx, val_idx, test_idx = patientwise_split(base_ds)

    # 3) Print & save stats for each split
    subset_stats(base_ds, train_idx, "Train")
    subset_stats(base_ds, val_idx, "Val")
    subset_stats(base_ds, test_idx, "Test")

    # 4) Create subsets and loaders
    train_ds = Subset(base_ds, train_idx)
    val_ds = Subset(base_ds, val_idx)
    test_ds = Subset(base_ds, test_idx)

    train_dl = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_VAL, shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=BATCH_VAL, shuffle=False, collate_fn=collate_fn)

    # 5) Model, optimizer
    model = SimpleCRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_params = count_parameters(model)
    print(f"\nTrainable parameters: {n_params:,}")

    # save model info
    with open(os.path.join(RESULT_DIR, "model_info.txt"), "w") as f:
        f.write(f"Trainable parameters: {n_params}\n")

    # 6) Training loop
    acc_hist = {"train": [], "val": [], "test": []}
    f1_hist = {"train": [], "val": [], "test": []}

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")

        tr_loss, tr_acc, tr_f1, _, _, _ = run_epoch(model, train_dl, optimizer)
        val_loss, val_acc, val_f1, _, _, _ = run_epoch(model, val_dl, optimizer=None)
        te_loss, te_acc, te_f1, _, _, _ = run_epoch(model, test_dl, optimizer=None)

        acc_hist["train"].append(tr_acc)
        acc_hist["val"].append(val_acc)
        acc_hist["test"].append(te_acc)

        f1_hist["train"].append(tr_f1)
        f1_hist["val"].append(val_f1)
        f1_hist["test"].append(te_f1)

        print(f"  Train: loss={tr_loss:.3f} acc={tr_acc:.3f} micro-F1={tr_f1:.3f}")
        print(f"  Val:   loss={val_loss:.3f} acc={val_acc:.3f} micro-F1={val_f1:.3f}")
        print(f"  Test:  loss={te_loss:.3f} acc={te_acc:.3f} micro-F1={te_f1:.3f}")

    # 7) Save per-epoch metrics to CSV
    metrics_df = pd.DataFrame({
        "epoch": np.arange(1, EPOCHS + 1),
        "train_acc": acc_hist["train"],
        "val_acc": acc_hist["val"],
        "test_acc": acc_hist["test"],
        "train_micro_f1": f1_hist["train"],
        "val_micro_f1": f1_hist["val"],
        "test_micro_f1": f1_hist["test"],
    })
    metrics_df.to_csv(os.path.join(RESULT_DIR, "metrics_per_epoch.csv"), index=False)

    # 8) Final evaluation on test for plots
    _, _, _, y_test, y_pred_test, y_prob_test = run_epoch(model, test_dl, optimizer=None)

    # Accuracy curves
    plot_curves(
        acc_hist,
        "Accuracy vs Epoch",
        "Accuracy",
        savepath=os.path.join(RESULT_DIR, "accuracy_curves.png")
    )

    # micro-F1 curves
    plot_curves(
        f1_hist,
        "Micro-F1 vs Epoch",
        "Micro-F1",
        savepath=os.path.join(RESULT_DIR, "micro_f1_curves.png")
    )

    # Confusion matrix
    plot_confusion(
        y_test,
        y_pred_test,
        LABELS,
        title="Test confusion matrix",
        savepath=os.path.join(RESULT_DIR, "confusion_matrix_test.png")
    )

    # ROC + AUC
    plot_multiclass_roc(
        y_test,
        y_prob_test,
        LABELS,
        title="Test ROC curves",
        savepath=os.path.join(RESULT_DIR, "roc_test.png")
    )


if __name__ == "__main__":
    main()
