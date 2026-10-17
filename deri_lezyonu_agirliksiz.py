# Deri Lezyonu — Deney A: AĞIRLIKSIZ kayıp (karşılaştırma için)
# Google Colab'da çalıştırmak için: Çalışma zamanı > Çalışma zamanı türünü değiştir > GPU
#
# 1) Kaggle'dan veri seti indirme (Colab hücresinde, başına ! koyarak):
#    !pip install -q kaggle
#    (kaggle.json dosyanı yükle: Kaggle > Settings > API > Create New Token)
#    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#    !kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
#    !unzip -q skin-cancer-mnist-ham10000.zip -d ham10000

import os, glob
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_DIR = "ham10000"
IMG_SIZE = 224
BATCH = 32
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 1. Meta veriyi oku ve ikili etikete çevir ----------
meta = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))

# Kötü huylu: melanom (mel), bazal hücreli karsinom (bcc), aktinik keratoz (akiec)
MALIGNANT = {"mel", "bcc", "akiec"}
meta["label"] = meta["dx"].apply(lambda d: 1 if d in MALIGNANT else 0)

# Görsel dosya yollarını bul (iki klasöre bölünmüş durumda)
paths = {os.path.splitext(os.path.basename(p))[0]: p
         for p in glob.glob(os.path.join(DATA_DIR, "**", "*.jpg"), recursive=True)}
meta["path"] = meta["image_id"].map(paths)
meta = meta.dropna(subset=["path"])

print("Sınıf dağılımı:\n", meta["label"].value_counts())

# ---------- 2. Eğitim / doğrulama / test ayrımı ----------
# ÖNEMLİ: Aynı lezyonun birden fazla fotoğrafı var. lesion_id'ye göre ayırmazsak
# aynı lezyon hem eğitimde hem testte olur ve sonuçlar yapay olarak yüksek çıkar.
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss.split(meta, groups=meta["lesion_id"]))
train_df, temp_df = meta.iloc[train_idx], meta.iloc[temp_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["lesion_id"]))
val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]
print(f"Eğitim: {len(train_df)}  Doğrulama: {len(val_df)}  Test: {len(test_df)}")

# ---------- 3. Veri yükleyici ve veri artırma ----------
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), norm,
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), norm,
])

class LesionDS(Dataset):
    def __init__(self, df, tf):
        self.df, self.tf = df.reset_index(drop=True), tf
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), int(r["label"])

train_dl = DataLoader(LesionDS(train_df, train_tf), BATCH, shuffle=True, num_workers=2)
val_dl   = DataLoader(LesionDS(val_df, eval_tf), BATCH, num_workers=2)
test_dl  = DataLoader(LesionDS(test_df, eval_tf), BATCH, num_workers=2)

# ---------- 4. Model: ImageNet ile önceden eğitilmiş MobileNetV2 ----------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(DEVICE)

# Sınıf dengesizliği: azınlık sınıfına (kötü huylu) daha yüksek ağırlık
counts = train_df["label"].value_counts().sort_index().values
weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss()  # DENEY A: sınıf ağırlığı YOK
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------- 5. Eğitim ve değerlendirme ----------
def evaluate(dl):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for x, y in dl:
            p = torch.softmax(model(x.to(DEVICE)), 1)[:, 1]
            probs += p.cpu().tolist(); ys += y.tolist()
    ys, probs = np.array(ys), np.array(probs)
    return ys, probs, (probs >= 0.5).astype(int)

best_auc = 0
for epoch in range(EPOCHS):
    model.train(); total = 0
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward(); optimizer.step()
        total += loss.item() * len(y)
    ys, probs, _ = evaluate(val_dl)
    auc = roc_auc_score(ys, probs)
    print(f"Epoch {epoch+1}/{EPOCHS}  kayıp={total/len(train_df):.4f}  doğrulama AUC={auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "agirliksiz_model.pt")

# ---------- 6. Test sonuçları (rapora girecek sayılar) ----------
model.load_state_dict(torch.load("agirliksiz_model.pt"))
ys, probs, preds = evaluate(test_dl)
print("\nTest AUC:", round(roc_auc_score(ys, probs), 4))
print(classification_report(ys, preds, target_names=["İyi huylu", "Kötü huylu"]))
print("Karmaşıklık matrisi:\n", confusion_matrix(ys, preds))
