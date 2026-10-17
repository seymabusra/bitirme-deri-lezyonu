# Proje Günlüğü

## 2026-10-03
- GitHub reposu açıldı, .gitignore ayarlandı
- Kaggle API anahtarı alındı
- HAM10000 indirildi (10015 görsel, 7470 lezyon)
- Sınıf dağılımı çıkarıldı: nv %67, kötü huylu toplam ~%19,5
- Grafik: figures/sinif_dagilimi.png, not defteri: notebooks/hafta1_veri.ipynb
- Sorun: yok

## 2026-10-03 (akşam)
- Yaş/cinsiyet dağılımı ve sınıf başına örnek görseller çıkarıldı
- İkili dağılım: %80,5 iyi huylu / %19,5 kötü huylu
- Görsellerin ~%45'i tekrar eden lezyonlara ait -> lesion_id ile bölme şart
- Yaş: çoğunluk 35-85, gençlerde az veri (sınırlılık olarak yazılacak)
- Görsellerde kıl, cetvel izi ve siyah kenar var (Grad-CAM'de kontrol edilecek)
- Grafikler: figures/yas_cinsiyet.png, figures/ornek_gorseller.png
- Sorun: indirilen not defteri "(1)" ekiyle kaydedildi, eskisi silinip yeniden adlandırıldı

## 2026-10-03 (gece)
- MobileNetV2 başlangıç modeli eğitildi (10 epoch, Colab T4 GPU, ~20 dk)
- En iyi doğrulama AUC: 0,9126 (epoch 7); sonrasında hafif ezberleme başladı
- Test AUC: 0,9181
- Kötü huylu: duyarlılık 0,76, kesinlik 0,62 -> 318 kötü huyludan 76'sı kaçırıldı
- Doğruluk %85, ama her şeye "iyi huylu" diyen model bile ~%79 alır (doğruluk yanıltıcı)
- Model: models/en_iyi_model.pt (GitHub'a yüklenmedi), sonuçlar: results/mobilenet_baseline.txt
- Sonraki adım: 3. haftada eşik ayarıyla duyarlılığı 0,85'e çıkarmak
- Sorun: Colab oturum kapatırken "Failed to unassign endpoint" hatası verdi, zararsızdı

## 2026-10-04 (Hafta 2)
- Bölme doğrulandı: 3 küme arasında ortak lezyon yok, CSV'ler results/ altında
- Veri artırma görseli: figures/augmentation.png
- Deney A (ağırlıksız): test AUC 0,9225, duyarlılık 0,69, 100 kanser kaçırıldı
- Deney B ile AUC aynı, fark karar sınırında: ağırlıklı model 24 kanser daha yakalıyor, 59 yanlış alarm daha veriyor
- Sonuç: ağırlıklı model kullanılacak; eşik ayarı 3. haftada
- Rapor taslağı oluşturuldu, Bölüm 5.2 dolduruldu

## 2026-10-04 (Hafta 3)
- Kayıtlı modelle tahminler yeniden üretildi (AUC'ler birebir aynı: 0,9126 / 0,9181)
- Tahminler kaydedildi: results/val_tahmin.csv, results/test_tahmin.csv
- Eşik doğrulamada seçildi: 0,35 (doğrulama duyarlılığı 0,867)
- Testte eşik 0,35: duyarlılık 0,76 -> 0,83, kaçırılan 76 -> 54, yanlış alarm 151 -> 208
- Testte hedef 0,85'in biraz altında kaldı; eşik test sonucuna göre değiştirilmedi
- Hata analizi: kaçırılan melanomlar nevüse benziyor, model yanlışlarda da çok emin (kalibrasyon önerisi)
- Grafikler: figures/esik_taramasi.png, figures/roc_test.png, figures/hatali_tahminler.png
- Rapor: Bölüm 5.3 ve 5.5 dolduruldu

## 2026-10-06
- Hafta 3 dosyaları repoya eklendi, hafta3.ipynb çıktılarıyla yeniden çalıştırılıp indirildi
- Hocaya 3 projeyi tanıtan kısa sunum hazırlandı
- Sorun: Colab aynı not defteri iki sekmede açık olduğu için "uzaktan değişiklik" uyarısı verdi, güncel sürüm kaydedildi
- Sonraki adım: Hafta 4 - ResNet-50 ve ince ayar karşılaştırması
## 2026-10-07 (Hafta 4 - 1. oturum)
- Hoca deri lezyonu projesini onayladı
- Hafta 4 için deney fonksiyonu yazıldı: hafta4_deney.py (tek satırla deney çalıştırma)
- MobileNetV2, sadece son katman (2.562 parametre): test AUC 0,878, eşik 0,40, duyarlılık 0,85, yanlış alarm 295, 13 dk
- ResNet-50, sadece son katman (4.098 parametre): test AUC 0,882, eşik 0,45, duyarlılık 0,86, yanlış alarm 310, 14 dk
- Sonuç: sadece son katmanı eğitmek açıkça daha kötü (AUC 0,92 -> 0,88); ImageNet özellikleri dermoskopi için tek başına yetmiyor
- Son katman modelleri birkaç kanser daha yakalıyor ama ~90-100 fazla yanlış alarm veriyor; karşılaştırmada AUC esas alınacak
- Süre 20 -> 13-14 dk'ya düştü; darboğaz model değil görsel okuma
- Sonuçlar: results/deneyler.csv
- Sonraki adım: ResNet-50 tüm ağ (s3), yaklaşık 30-50 dk
## 2026-10-08 (Hafta 4 - 2. oturum)
- ResNet-50, tüm ağ (23,5 milyon parametre): doğrulama AUC 0,925, test AUC 0,914, eşik 0,50, duyarlılık 0,83, yanlış alarm 198, 17 dk
- ResNet-50 ve MobileNetV2 (tüm ağ) arasında anlamlı fark yok; doğrulama ve test sıralaması ters
- Nihai model: MobileNetV2 tüm ağ + eşik 0,35 (10 kat daha az parametre, daha küçük ve hızlı)
- results/deneyler.csv üç deneyle birleştirildi; rapor Bölüm 5.4 dolduruldu
- Sonraki adım: Hafta 5 - Grad-CAM

## 2026-10-09 (Hafta 5)
- Grad-CAM uygulandı (MobileNetV2, son evrişim bloğu, eşik 0,35)
- Kenar oranı = dikkatin görüntünün dış %15 çerçevesine düşen payı (rastgele dağılımda ~0,51)
- Kötü huylu sınıfı için: doğru kötü huylu 0,13, yanlış alarm 0,18, kaçırılan 0,45, doğru iyi huylu 0,89 (zayıf harita, yorumlanmadı)
- Tahmin edilen sınıf için: tüm gruplarda 0,08-0,18 -> model kararlarını lezyona bakarak veriyor
- Kaçırılan kötü huylularda model lezyona en yoğun bakıyor (0,08) ama iyi huylu diyor; hata yan unsurlardan değil, nevüse benzerlikten
- Kıl, cetvel, siyah kenar üzerinden kısa yol öğrenme gözlenmedi
- Grafikler: figures/gradcam_*.png (6 dosya), sonuçlar: results/gradcam_kenar.csv
- Rapor: Bölüm 6 dolduruldu (Şekil 10, 11, Tablo 7), Özet tamamlandı
- Sorun: İndirilen dosyalar "(1)" ekiyle kaydedildi, csv yanlışlıkla figures/'a da düşmüştü; düzeltildi
- Sonraki adım: Hafta 6 - Streamlit arayüzü
## 2026-10-09 / 2026-10-11 (Hafta 6 - arayüz)
- Bilgisayarda Python 3.14 ile sanal ortam (.venv) kuruldu; PyTorch CPU sürümü yüklendi
- İlk arayüz Streamlit ile yazıldı (app/app.py), çalıştı ama görünümü yetersiz bulundu
- Streamlit bırakıldı; kendi web sitesi yazıldı:
  - app/server.py: FastAPI sunucusu, modeli ve Grad-CAM'i çalıştırıyor (/api/analiz)
  - app/web/index.html: tek sayfalık ön yüz (HTML/CSS/JavaScript)
- Site özellikleri: sürükle-bırak yükleme, dermoskop merceği görünümü, tarama ve halka animasyonu,
  eşik %35 işareti, ısı haritası açma/kapama ve yoğunluk ayarı, iki sınıflı Grad-CAM,
  etkileşimli eşik grafiği (results/esik_taramasi.csv), Şekil 11 ile hata açıklaması
- Çalıştırma: python app/server.py -> http://127.0.0.1:8000
- Sorun: pip komutu kopyalanırken iki satıra bölündü; requirements.txt yanlış klasördeydi, düzeltildi
- Sonraki adım: app/ornekler/ için demo görselleri, son testler

### 2026-10-11 — Demo görselleri ve gerçek model testi

- app/ornekler/: test kümesinden 6 demo görseli eklendi (doğru tespit x2, doğru iyi huylu x2, kaçırılan, yanlış alarm). Seçim kodu: notebooks/hafta6_ornekler.ipynb
- Site gerçek modelle test edildi: melanom %84,7 (doğru tespit), kaçırılan melanom %14,4, yanlış alarm %67,6. Sonuçlar Colab ile aynı.
- Eşik grafiği esik_taramasi.csv ile tutarlı (0,35'te duyarlılık %86,7, 37 kaçırılan, 253 yanlış alarm).
- Çalıştırma: .venv\Scripts\activate, sonra python app/server.py
- Sonraki adım: Hafta 7 — rapor (web arayüzü + Sonuç bölümü), poster, sunum

## 2026-10-11 (Hafta 7 - rapor)
- Rapora Bölüm 7 Web Arayüzü eklendi: mimari, uç noktalar (Tablo 8), kullanım (Şekil 12), arayüz doğrulaması (Tablo 9, Şekil 13)
- Tartışma (Bölüm 8) genişletildi: eşiğin bedeli (22 ek tespit için 57 ek yanlış alarm), hataların ortak özellikleri, arayüzün sınırlılıkları
- Sonuç ve gelecek çalışmalar (Bölüm 9) yazıldı
- figures/web_arayuz.png ve figures/web_uc_durum.png eklendi

