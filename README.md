# Görsel Sınıflandırıcı Demo (Animals-10)
DATASETİ GİTHUBA YÜKLEMEDE SIKINTI YAŞADIM KODUM SIKINTISIZ ÇALIŞIYOR EĞİTİMİ TAMAMLANDI

Bu proje, PyTorch kullanılarak geliştirilmiş basit bir **görüntü sınıflandırma uygulamasıdır**. Uygulama, [Animals-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) kullanılarak eğitilen bir derin öğrenme modeliyle çalışır. Arayüz olarak Streamlit tercih edilmiştir.

## 🔍 Özellikler

- 10 hayvan sınıfı: `cane`, `cavallo`, `elefante`, `farfalla`, `gallina`, `gatto`, `mucca`, `pecora`, `ragno`, `scoiattolo`
- Derin öğrenme tabanlı sınıflandırma modeli (PyTorch)
- Streamlit ile kullanıcı dostu görsel arayüz
- Model dosyası `.pth` formatında saklanır

## 📁 Proje Yapısı

```
gorsel-siniflandirici-demo/
│
├── app.py                   # Streamlit arayüzü
├── train.py                 # Model eğitimi
├── model.py                 # Model mimarisi
├── dataloader.py            # Dataloader tanımı
├── prepare_data.py          # Veri ön işleme scripti
├── model.pth                # Eğitilmiş model dosyası
├── requirements.txt         # Gereken kütüphaneler
├── dataset/                 # Eğitim verileri (Animals-10)
└── .gitignore               # Git ayarları
```

## 🚀 Kurulum ve Çalıştırma

1. Ortamı oluştur:

```bash
python -m venv venv
source venv/bin/activate  # (MacOS/Linux)
venv\Scripts\activate     # (Windows)
```

2. Gerekli kütüphaneleri yükle:

```bash
pip install -r requirements.txt
```

3. Streamlit uygulamasını başlat:

```bash
streamlit run app.py
```

## 🧠 Eğitim Sonuçları

Model yaklaşık olarak:
- Eğitim Doğruluğu: %85+
- Doğrulama Doğruluğu: %58 civarında

Not: Daha yüksek doğruluk için daha büyük model mimarileri ve daha uzun eğitim gerekebilir.

## 📷 Örnek Ekran Görüntüsü

> `app.py` çalıştırıldığında ortaya çıkan arayüzü buraya bir ekran görüntüsü olarak ekleyebilirsiniz.

## 📌 Gereksinimler

- Python 3.8+
- PyTorch
- Streamlit
- torchvision
- PIL

## 👩‍💻 Geliştirici

**Merve Kahya**  

---

> Bu proje, eğitim amaçlı geliştirilmiş bir demodur. Daha gelişmiş versiyonları için model optimizasyonu ve veri artırma yöntemleri eklenebilir.
