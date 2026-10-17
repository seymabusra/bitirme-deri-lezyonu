# Deri Lezyonu — Hafta 5: Grad-CAM ile açıklanabilirlik
# Gerekenler (soldaki panelden yükle): kaggle.json, en_iyi_model.pt, test_tahmin.csv
# Bu hücre sadece hazırlık yapar. Sonra ayrı hücrelerde:
#   ornekleri_ciz()     -> 4 grup için orijinal + ısı haritası görselleri
#   ozet = kenar_analizi()  -> modelin görüntü kenarlarına ne kadar baktığının sayısal kontrolü

import os, glob
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ESIK = 0.35  # Hafta 3'te doğrulama kümesinde seçilen eşik
print("Cihaz:", DEVICE)

# ---------- Nihai model: MobileNetV2, tüm ağ, ağırlıklı kayıp ----------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("en_iyi_model.pt", map_location=DEVICE))
model = model.to(DEVICE).eval()

# Grad-CAM, son evrişim bloğunun çıktısına bakar; hedef sınıf 1 = "kötü huylu"
cam = GradCAM(model=model, target_layers=[model.features[-1]])
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def isi_haritasi(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    x = norm(transforms.functional.to_tensor(img)).unsqueeze(0).to(DEVICE)
    mask = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
    return rgb, mask

# ---------- Hafta 3'te kaydedilen test tahminleri ----------
tp = pd.read_csv("test_tahmin.csv")
yollar = {os.path.splitext(os.path.basename(p))[0]: p
          for p in glob.glob("ham10000/**/*.jpg", recursive=True)}
tp["path"] = tp["image_id"].map(yollar)
tp["pred"] = (tp.prob >= ESIK).astype(int)
print("Test görseli:", len(tp), "| yolu bulunamayan:", tp.path.isna().sum())

gruplar = {
    "Doğru tespit edilen kötü huylu": tp[(tp.label == 1) & (tp.pred == 1)].nlargest(4, "prob"),
    "Doğru tespit edilen iyi huylu": tp[(tp.label == 0) & (tp.pred == 0)].nsmallest(4, "prob"),
    "Kaçırılan kötü huylu": tp[(tp.label == 1) & (tp.pred == 0)].nsmallest(4, "prob"),
    "Yanlış alarm": tp[(tp.label == 0) & (tp.pred == 1)].nlargest(4, "prob"),
}
DOSYALAR = ["gradcam_dogru_kotu.png", "gradcam_dogru_iyi.png",
            "gradcam_kacirilan.png", "gradcam_yanlis_alarm.png"]

def ornekleri_ciz():
    ozet_kareler = []
    for (baslik, df), dosya in zip(gruplar.items(), DOSYALAR):
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
        satir = []
        for j, (_, r) in enumerate(df.iterrows()):
            rgb, mask = isi_haritasi(r.path)
            ust = show_cam_on_image(rgb, mask, use_rgb=True)
            axes[0, j].imshow(rgb)
            axes[0, j].set_title(f"{r.dx} | olasılık {r.prob:.2f}", fontsize=10)
            axes[1, j].imshow(ust)
            satir.append(ust)
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"{baslik} (üst: orijinal, alt: Grad-CAM)", fontsize=13)
        plt.tight_layout(); plt.savefig(dosya, dpi=120); plt.show()
        ozet_kareler.append((baslik, satir))

    # Poster için tek görselde özet: her satır bir grup
    fig, axes = plt.subplots(4, 4, figsize=(11, 12))
    for i, (baslik, satir) in enumerate(ozet_kareler):
        for j in range(4):
            axes[i, j].axis("off")
            if j < len(satir):
                axes[i, j].imshow(satir[j])
        axes[i, 0].set_title(baslik, fontsize=11, loc="left")
    plt.tight_layout(); plt.savefig("gradcam_ozet.png", dpi=120); plt.show()

# ---------- Sayısal kontrol: model görüntünün kenarlarına ne kadar bakıyor? ----------
# Kenar = görüntünün dış %15'lik çerçevesi (alanın yaklaşık %51'i).
# Isı haritası düzgün dağılsaydı oran ~0,51 olurdu; daha düşük değer modelin merkeze
# (lezyonun genelde bulunduğu yere) odaklandığını, daha yüksek değer kenarlardaki
# siyah çerçeve, cetvel gibi unsurlara baktığını düşündürür.
def kenar_orani(mask, pay=0.15):
    h, w = mask.shape
    a, b = int(h * pay), int(w * pay)
    return float(1 - mask[a:h - a, b:w - b].sum() / (mask.sum() + 1e-8))

def kenar_analizi():
    oranlar = []
    for i, (_, r) in enumerate(tp.iterrows()):
        oranlar.append(kenar_orani(isi_haritasi(r.path)[1]))
        if (i + 1) % 300 == 0:
            print(f"{i + 1}/{len(tp)} görsel işlendi")
    tp["kenar_orani"] = oranlar
    tp["grup"] = np.select(
        [(tp.label == 1) & (tp.pred == 1), (tp.label == 0) & (tp.pred == 0),
         (tp.label == 1) & (tp.pred == 0), (tp.label == 0) & (tp.pred == 1)],
        ["Doğru kötü huylu", "Doğru iyi huylu", "Kaçırılan", "Yanlış alarm"], default="?")
    ozet = tp.groupby("grup")["kenar_orani"].agg(["count", "mean", "median"]).round(3)
    print("\nDüzgün dağılımda beklenen kenar oranı: 0.51")
    print(ozet)
    tp.drop(columns=["path"]).to_csv("gradcam_kenar.csv", index=False)
    return ozet

print("Hazır. Şimdi: ornekleri_ciz()")
