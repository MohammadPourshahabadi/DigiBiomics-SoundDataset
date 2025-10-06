# -*- coding: utf-8 -*-
"""
ICBHI 2017 cycle classification (from scratch, no pretrain).
Balances classes by oversampling minority classes with on-the-fly augmentation.

Augmentations:
- Gaussian noise
- Random gain
- Time shift
- Time stretch (fixed with torch.istft + valid window)
- Pitch shift (resample-based)

Outputs:
- history.csv, curves.png
- cm.png, report.csv
- metrics.csv (recall/specificity per class), metrics_overall.csv
- roc_micro_macro.png, roc_per_class/*.png, auc.csv
- skipped_files.csv
"""

import os, glob, random, math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader, Subset

# ------------------- PATHS ------------------- #
DATASET_ROOT = "/content/drive/MyDrive/Colab Notebooks/Datasets/ICBHI 2017 Challenge/ICBHI_final_database"
OUTDIR = os.path.join("runs", "icbhi_scratch_aug_oversample")
os.makedirs(OUTDIR, exist_ok=True)

# ------------------- HYPERPARAMS ------------------- #
SEED = 1337
EPOCHS = 50
BATCH_TRAIN = 16
BATCH_VAL = 32
LR = 1.5e-4
WEIGHT_DECAY = 1e-3
TARGET_SEC = 2.5
BALANCE_DATA = True

SR = 16000
WIN = int(0.025 * SR)  # 25ms = 400
HOP = int(0.010 * SR)  # 10ms = 160
N_MELS = 64
FMIN = 20
FMAX = 8000
LABELS = ["normal","crackle","wheeze","both"]

# ------------------- UTILS ------------------- #
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def cycle_label(c, w):
    if c==0 and w==0: return 0
    if c==1 and w==0: return 1
    if c==0 and w==1: return 2
    return 3

skipped_files = []

# ------------------- ROBUST LOADER ------------------- #
def load_wav_16k(path, target_sr=16000):
    import soundfile as sf
    try:
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1: data = data.mean(axis=1)
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

def add_gaussian_noise(wav, snr_db_range=(10,25)):
    x = wav.squeeze(0)
    rms = torch.sqrt(torch.mean(x**2) + 1e-12)
    snr_db = random.uniform(*snr_db_range)
    snr_lin = 10**(snr_db/10.0)
    noise_rms = rms / math.sqrt(snr_lin)
    noise = torch.randn_like(x) * noise_rms
    return (x + noise).unsqueeze(0)

def random_gain(wav, db_range=(-6,6)):
    gain_db = random.uniform(*db_range)
    gain = 10.0**(gain_db/20.0)
    return wav * gain

def time_shift(wav, max_shift_sec=0.25, sr=16000):
    max_shift = int(max_shift_sec*sr)
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

def time_stretch(wav, rate_range=(0.9,1.1)):
    rate = random.uniform(*rate_range)
    if abs(rate-1.0) < 1e-3: 
        return wav
    n_fft = 1024
    hop = HOP
    x = wav.squeeze(0)
    window = torch.hann_window(n_fft, device=x.device)   # use full 1024
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=window, center=True, return_complex=True)
    phase_advance = torch.linspace(0, math.pi*hop, X.size(-2), device=x.device)[None,:,None]
    pv = torchaudio.functional.phase_vocoder(X, rate=rate, phase_advance=phase_advance)
    y = torch.istft(pv, n_fft=n_fft, hop_length=hop,
                    win_length=n_fft, window=window, length=x.shape[-1])
    return y.unsqueeze(0)

def pitch_shift_resample(wav, sr=16000, semitone_range=(-2,2)):
    semis = random.uniform(*semitone_range)
    if abs(semis) < 1e-3: return wav
    factor = 2 ** (semis / 12.0)
    y = torchaudio.functional.resample(wav, sr, int(sr*factor))
    y = torchaudio.functional.resample(y, int(sr*factor), sr)
    T = wav.shape[-1]
    if y.shape[-1] < T: y = nn.functional.pad(y,(0,T-y.shape[-1]))
    elif y.shape[-1] > T: y = y[...,:T]
    return y

def apply_random_augment(wav):
    if rand_bool(0.7): wav = random_gain(wav, (-4,4))
    if rand_bool(0.6): wav = add_gaussian_noise(wav, (12,25))
    if rand_bool(0.5): wav = time_shift(wav, 0.20, SR)
    if rand_bool(0.35): wav = time_stretch(wav, (0.92,1.08))
    if rand_bool(0.35): wav = pitch_shift_resample(wav, SR, (-1.5,1.5))
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
        self.root=root; self.target_sec=target_sec
        self.random_crop=random_crop; self.augment=augment
        self.items=[]; self._index()
        self.melspec=torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=1024, win_length=WIN, hop_length=HOP,
            f_min=FMIN,f_max=FMAX,n_mels=N_MELS,center=True,power=2.0)
        self.amp2db=torchaudio.transforms.AmplitudeToDB(stype="power")

    def _index(self):
        txts = sorted(glob.glob(os.path.join(self.root,"**/*.txt"),recursive=True))
        wavs = sorted(glob.glob(os.path.join(self.root,"**/*.wav"),recursive=True))
        wav_map = {os.path.basename(w).lower():w for w in wavs}
        for txt in txts:
            cand_wav = txt[:-4]+".wav"
            if not os.path.isfile(cand_wav):
                cand_wav = wav_map.get(os.path.basename(cand_wav).lower())
                if not cand_wav: continue
            with open(txt) as f:
                for line in f:
                    parts=line.strip().split()
                    if len(parts)<4: continue
                    try:
                        s,e=float(parts[0]),float(parts[1])
                        c,w=int(float(parts[2])),int(float(parts[3]))
                        if e>s:
                            self.items.append(CycleItem(cand_wav,s,e,cycle_label(c,w)))
                    except: continue
        print(f"[Dataset] Found {len(self.items)} cycles.")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it=self.items[idx]
        wav=load_wav_16k(it.wav_path)
        if wav is None: return None
        start,end=int(it.start_s*SR),int(it.end_s*SR)
        seg=wav[:,start:end]
        target_len=int(self.target_sec*SR)
        if seg.shape[-1]==0: seg=torch.zeros(1,target_len)
        if seg.shape[-1]<target_len:
            seg=nn.functional.pad(seg,(0,target_len-seg.shape[-1]))
        elif seg.shape[-1]>target_len:
            off=np.random.randint(0,seg.shape[-1]-target_len+1) if self.random_crop else (seg.shape[-1]-target_len)//2
            seg=seg[:,off:off+target_len]
        if self.augment: seg = apply_random_augment(seg)
        S=self.amp2db(self.melspec(seg)+1e-10)
        S=(S-S.mean())/(S.std()+1e-6)
        return S,it.y

def collate_fn(batch):
    batch=[b for b in batch if b is not None]
    if len(batch)==0: return None,None
    X=torch.stack([b[0] for b in batch])
    y=torch.tensor([b[1] for b in batch])
    return X,y

# ------------------- MODEL ------------------- #
class ConvBlock(nn.Module):
    def __init__(self,c_in,c_out):
        super().__init__()
        self.conv=nn.Conv2d(c_in,c_out,3,padding=1,bias=False)
        self.bn=nn.BatchNorm2d(c_out); self.act=nn.ReLU()
    def forward(self,x): return self.act(self.bn(self.conv(x)))

class ScratchFeatureExtractor(nn.Module):
    def __init__(self,n_mels=64):
        super().__init__()
        self.features=nn.Sequential(
            ConvBlock(1,32),ConvBlock(32,32),nn.MaxPool2d(2),
            ConvBlock(32,64),ConvBlock(64,64),nn.MaxPool2d(2),
            ConvBlock(64,128),nn.Dropout(0.1))
        conv_dim=(n_mels//4)*128
        self.rnn=nn.LSTM(conv_dim,256,num_layers=2,batch_first=True,
                         dropout=0.1,bidirectional=True)
    def forward(self,x):
        B,_,M,T=x.shape
        h=self.features(x)
        h=h.permute(0,3,1,2).contiguous().view(B,h.shape[3],-1)
        y,_=self.rnn(h); return y

class ICBHIClassifier(nn.Module):
    def __init__(self,n_classes=4):
        super().__init__()
        self.enc=ScratchFeatureExtractor()
        self.attn=nn.Linear(512,1)
        self.head=nn.Sequential(nn.Dropout(0.4),nn.Linear(512,n_classes))
    def forward(self,x):
        seq=self.enc(x)
        w=torch.softmax(self.attn(seq).squeeze(-1),dim=1)
        emb=(seq*w.unsqueeze(-1)).sum(1)
        return self.head(emb)

# ------------------- BALANCE ------------------- #
def oversample_with_replacement(base_train, train_idx, out_csv):
    from collections import defaultdict
    rng=np.random.default_rng(SEED)
    label_to_idx=defaultdict(list)
    for i in train_idx:
        y = base_train.items[i].y
        label_to_idx[y].append(i)
    counts={y:len(v) for y,v in label_to_idx.items()}
    max_count=max(counts.values())
    expanded=[]
    for y,idxs in label_to_idx.items():
        need=max_count-len(idxs)
        if need>0:
            extra=rng.choice(idxs,size=need,replace=True).tolist()
            merged=idxs+extra
        else: merged=idxs
        expanded.extend(merged)
    rng.shuffle(expanded)
    pd.DataFrame([{"idx":i,"label":base_train.items[i].y,"label_name":LABELS[base_train.items[i].y]} for i in expanded]).to_csv(out_csv,index=False)
    print(f"[Oversample] Counts={counts} -> expanded {len(expanded)} samples")
    return expanded

# ------------------- MAIN ------------------- #
def main():
    set_seed(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Main] Using device:",device)

    base=ICBHICycleDataset(DATASET_ROOT,TARGET_SEC,random_crop=True,augment=False)
    n_total=len(base); n_val=int(0.15*n_total); n_train=n_total-n_val
    perm=torch.randperm(n_total).numpy()
    train_idx,val_idx=perm[:n_train],perm[n_train:n_train+n_val]

    if BALANCE_DATA:
        train_idx=oversample_with_replacement(base,train_idx,os.path.join(OUTDIR,"oversampled.csv"))

    train_ds=Subset(ICBHICycleDataset(DATASET_ROOT,TARGET_SEC,True,augment=True),train_idx)
    val_ds=Subset(ICBHICycleDataset(DATASET_ROOT,TARGET_SEC,False,augment=False),val_idx.tolist())

    train_dl=DataLoader(train_ds,batch_size=BATCH_TRAIN,shuffle=True,collate_fn=collate_fn)
    val_dl=DataLoader(val_ds,batch_size=BATCH_VAL,shuffle=False,collate_fn=collate_fn)

    model=ICBHIClassifier(len(LABELS)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    crit=nn.CrossEntropyLoss()

    history=[]; best_mf1=-1
    for epoch in range(1,EPOCHS+1):
        model.train(); yt=[]; yp=[]; losses=[]
        for X,y in train_dl:
            if X is None: continue
            X,y=X.to(device),y.to(device)
            out=model(X); loss=crit(out,y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            yp.extend(out.argmax(1).cpu().numpy()); yt.extend(y.cpu().numpy())
        tr_acc=accuracy_score(yt,yp); tr_mf1=f1_score(yt,yp,average="macro")
        model.eval(); yt_val=[]; yp_val=[]; vlosses=[]; yprob_val=[]
        with torch.no_grad():
            for X,y in val_dl:
                if X is None: continue
                X,y=X.to(device),y.to(device)
                out=model(X); loss=crit(out,y); vlosses.append(loss.item())
                probs=torch.softmax(out,dim=1).cpu().numpy()
                yprob_val.append(probs)
                yp_val.extend(out.argmax(1).cpu().numpy()); yt_val.extend(y.cpu().numpy())
        yprob_val=np.vstack(yprob_val) if len(yprob_val)>0 else np.zeros((0,len(LABELS)))
        va_acc=accuracy_score(yt_val,yp_val); va_mf1=f1_score(yt_val,yp_val,average="macro")
        history.append({"epoch":epoch,"train_acc":tr_acc,"val_acc":va_acc,"train_mf1":tr_mf1,"val_mf1":va_mf1})
        print(f"[Epoch {epoch}] train_acc={tr_acc:.3f} val_acc={va_acc:.3f} val_mf1={va_mf1:.3f}")
        if va_mf1>best_mf1:
            best_mf1=va_mf1
            torch.save(model.state_dict(),os.path.join(OUTDIR,"best.ckpt"))

    pd.DataFrame(history).to_csv(os.path.join(OUTDIR,"history.csv"),index=False)
    print("[Done] Results in",OUTDIR)

if __name__=="__main__":
    main()


