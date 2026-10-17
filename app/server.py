# Deri Lezyonu Ön Değerlendirme — web sunucusu (Hafta 6)
# Çalıştırma (reponun ana klasöründe, sanal ortam açıkken):  python app/server.py
# Tarayıcıda http://127.0.0.1:8000 kendiliğinden açılır.
#
# Gerekenler : models/en_iyi_model.pt
# İsteğe bağlı: app/ornekler/*.jpg|png        (demo görselleri)
#               results/esik_taramasi.csv     (eşik grafiği)
#               figures/gradcam_kacirilan_iki_sinif.png

import base64
import io
import threading
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

APP = Path(__file__).resolve().parent
KOK = APP.parent
MODEL_YOLU = KOK / "models" / "en_iyi_model.pt"
ORNEKLER = APP / "ornekler"
ESIK_CSV = KOK / "results" / "esik_taramasi.csv"
FIGURLER = KOK / "figures"
ESIK = 0.35  # doğrulama kümesinde seçilen karar eşiği (Hafta 3)
UZANTILAR = {".jpg", ".jpeg", ".png"}

if not MODEL_YOLU.exists():
    raise SystemExit(f"Model dosyası bulunamadı: {MODEL_YOLU}\n"
                     "en_iyi_model.pt dosyasını reponun models/ klasörüne koyun.")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(MODEL_YOLU, map_location="cpu"))
model.eval()
cam = GradCAM(model=model, target_layers=[model.features[-1]])
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
kilit = threading.Lock()

app = FastAPI(title="Deri Lezyonu Ön Değerlendirme")


def png_b64(rgb_uint8):
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def isi_rengi(maske):
    renkli = cv2.applyColorMap(np.uint8(255 * maske), cv2.COLORMAP_JET)
    return cv2.cvtColor(renkli, cv2.COLOR_BGR2RGB)


@app.post("/api/analiz")
async def analiz(dosya: UploadFile = File(...)):
    veri = await dosya.read()
    try:
        img = Image.open(io.BytesIO(veri)).convert("RGB").resize((224, 224))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Bu dosya bir görüntü olarak açılamadı. JPG veya PNG bir dermoskopi görüntüsü seçin.")
    x = NORM(transforms.functional.to_tensor(img)).unsqueeze(0)
    with kilit:
        with torch.no_grad():
            olasilik = torch.softmax(model(x), 1)[0, 1].item()
        karar = int(olasilik >= ESIK)
        m_kotu = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
        m_iyi = cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])[0]
    return {
        "olasilik": olasilik, "karar": karar, "esik": ESIK,
        "foto": png_b64(np.asarray(img)),
        "harita_karar": png_b64(isi_rengi(m_kotu if karar else m_iyi)),
        "harita_kotu": png_b64(isi_rengi(m_kotu)),
    }


@app.get("/api/ornekler")
def ornek_listesi():
    if not ORNEKLER.exists():
        return []
    return [p.name for p in sorted(ORNEKLER.iterdir()) if p.suffix.lower() in UZANTILAR]


@app.get("/ornekler/{ad}")
def ornek(ad: str):
    yol = (ORNEKLER / ad).resolve()
    if yol.parent != ORNEKLER.resolve() or yol.suffix.lower() not in UZANTILAR or not yol.exists():
        raise HTTPException(404)
    return FileResponse(yol)


@app.get("/api/esik")
def esik_tablosu():
    if not ESIK_CSV.exists():
        return []
    t = pd.read_csv(ESIK_CSV)
    t.columns = ["esik", "duyarlilik", "kesinlik", "yanlis_alarm", "kacirilan"]
    return JSONResponse(t.to_dict(orient="records"))


@app.get("/figurler/{ad}")
def figur(ad: str):
    yol = (FIGURLER / ad).resolve()
    if yol.parent != FIGURLER.resolve() or yol.suffix.lower() != ".png" or not yol.exists():
        raise HTTPException(404)
    return FileResponse(yol)


@app.get("/")
def ana_sayfa():
    return FileResponse(APP / "web" / "index.html")


if __name__ == "__main__":
    adres = "http://127.0.0.1:8000"
    print(f"\nUygulama hazır: {adres}\nKapatmak için bu pencerede Ctrl + C\n")
    threading.Timer(1.5, lambda: webbrowser.open(adres)).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
