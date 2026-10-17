# Deri Lezyonu — Hafta 4: Model ve ince ayar karşılaştırması
# Bu hücre her şeyi hazırlar (veri, bölme, fonksiyonlar). Eğitim YAPMAZ.
# Sonra ayrı hücrelerde şunlardan birini çalıştır:
#   s1 = deney("mobilenet", "son_katman")
#   s2 = deney("resnet50", "son_katman")
#   s3 = deney("resnet50", "tum_ag")

import os, glob, time, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix

DATA_DIR = "ham10000"
IMG_SIZE = 224
BATCH = 32
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Cihaz:", DEVICE)

# ---------- Veri ve bölme (önceki haftalarla birebir aynı) ----------
meta = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))
meta["label"] = meta["dx"].apply(lambda d: 1 if d in {"mel", "bcc", "akiec"} else 0)
paths = {os.path.splitext(os.path.basename(p))[0]: p
         for p in glob.glob(os.path.join(DATA_DIR, "**", "*.jpg"), recursive=True)}
meta["path"] = meta["image_id"].map(paths)
meta = meta.dropna(subset=["path"])

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss.split(meta, groups=meta["lesion_id"]))
train_df, temp_df = meta.iloc[train_idx], meta.iloc[temp_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["lesion_id"]))
val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]
print(f"Eğitim: {len(train_df)}  Doğrulama: {len(val_df)}  Test: {len(test_df)}")

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20), transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm])

class LesionDS(Dataset):
    def __init__(self, df, tf):
        self.df, self.tf = df.reset_index(drop=True), tf
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), int(r["label"])

train_dl = DataLoader(LesionDS(train_df, train_tf), BATCH, shuffle=True, num_workers=2)
val_dl = DataLoader(LesionDS(val_df, eval_tf), BATCH, num_workers=2)
test_dl = DataLoader(LesionDS(test_df, eval_tf), BATCH, num_workers=2)

# Ağırlıklı kayıp (Hafta 2'de seçtiğimiz yöntem)
counts = train_df["label"].value_counts().sort_index().values
class_weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(DEVICE)

# ---------- Model kurulumu ----------
def model_kur(ad, ayar):
    if ad == "mobilenet":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        if ayar == "son_katman":
            for p in m.parameters():
                p.requires_grad = False
        m.classifier[1] = nn.Linear(m.last_channel, 2)
    elif ad == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if ayar == "son_katman":
            for p in m.parameters():
                p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        raise ValueError("Model adı 'mobilenet' veya 'resnet50' olmalı")
    return m.to(DEVICE)

def tahmin(model, dl):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps += torch.softmax(model(x.to(DEVICE)), 1)[:, 1].cpu().tolist()
            ys += y.tolist()
    return np.array(ys), np.array(ps)

def esik_sec(y, p):
    # Hafta 3'teki kural: doğrulamada duyarlılığın >= 0,85 olduğu en yüksek eşik
    for t in np.round(np.arange(0.70, 0.19, -0.05), 2):
        if recall_score(y, (p >= t).astype(int)) >= 0.85:
            return float(t)
    return 0.20

def olc(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
    return {"duyarlılık": round(tp / (tp + fn), 3), "kesinlik": round(tp / (tp + fp), 3),
            "özgüllük": round(tn / (tn + fp), 3), "kaçırılan": int(fn), "yanlış alarm": int(fp)}

# ---------- Deney ----------
def deney(ad, ayar):
    torch.manual_seed(42)
    model = model_kur(ad, ayar)
    egitilen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lr = 1e-3 if ayar == "son_katman" else 1e-4
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    dosya = f"{ad}_{ayar}.pt"
    best_auc, best_ep, t0 = 0, 0, time.time()
    print(f"=== {ad} / {ayar} | eğitilen parametre: {egitilen:,} | lr = {lr}")

    for ep in range(EPOCHS):
        model.train()
        total = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        yv, pv = tahmin(model, val_dl)
        auc = roc_auc_score(yv, pv)
        print(f"Epoch {ep + 1}/{EPOCHS}  kayıp={total / len(train_df):.4f}  doğrulama AUC={auc:.4f}  ({(time.time() - t0) / 60:.1f} dk)")
        if auc > best_auc:
            best_auc, best_ep = auc, ep + 1
            torch.save(model.state_dict(), dosya)

    sure = (time.time() - t0) / 60
    model.load_state_dict(torch.load(dosya, map_location=DEVICE))
    yv, pv = tahmin(model, val_dl)
    yt, pt = tahmin(model, test_dl)
    esik = esik_sec(yv, pv)

    sonuc = {"model": ad, "ince ayar": ayar, "eğitilen parametre": egitilen,
             "en iyi epoch": best_ep, "doğrulama AUC": round(float(best_auc), 4),
             "test AUC": round(float(roc_auc_score(yt, pt)), 4), "süre (dk)": round(sure, 1),
             "seçilen eşik": esik}
    for k, v in olc(yt, pt, 0.5).items():
        sonuc[f"{k} @0.5"] = v
    for k, v in olc(yt, pt, esik).items():
        sonuc[f"{k} @eşik"] = v

    print("\n" + json.dumps(sonuc, ensure_ascii=False, indent=1))
    pd.DataFrame([sonuc]).to_csv("deneyler.csv", mode="a", index=False,
                                 header=not os.path.exists("deneyler.csv"))
    return sonuc

print("Hazır. Şimdi bir deney çalıştır, ör: s1 = deney('mobilenet', 'son_katman')")
