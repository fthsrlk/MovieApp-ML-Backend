# Film ve Dizi Öneri Sistemi - Makine Öğrenmesi Modülü

Bu modül, [Film ve Dizi Arşivim] uygulaması için kişiselleştirilmiş içerik önerileri oluşturan makine öğrenmesi bileşenidir.

## Özellikler

- İşbirlikçi filtreleme (Collaborative filtering)
- İçerik tabanlı filtreleme (Content-based filtering)
- Hibrit öneri algoritmaları
- REST API entegrasyonu
- TMDB veri işleme

## Kurulum

```bash
# Sanal ortam oluşturma (Opsiyonel)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Gerekliliklerin kurulumu
pip install -r requirements.txt
```

## Proje Yapısı

```
ml_recommendation_engine/
├── api/               # API entegrasyonu
│   ├── __init__.py
│   ├── app.py         # Flask uygulaması
│   └── routes.py      # API endpointleri
├── data/              # Veri işleme
│   ├── __init__.py
│   ├── loader.py      # Veri yükleme
│   └── preprocessor.py # Veri ön işleme
├── models/            # Öneri modelleri
│   ├── __init__.py
│   ├── collaborative.py # İşbirlikçi filtreleme
│   ├── content_based.py # İçerik tabanlı filtreleme
│   └── hybrid.py      # Hibrit model
├── __init__.py
├── config.py          # Yapılandırma ayarları
├── main.py            # Ana giriş noktası
├── README.md
└── requirements.txt
```

## Çalıştırma

```bash
# Geliştirme modu
python main.py --dev

# API sunucusu
python -m api.app
```

## API Kullanımı

### Öneri Alma

```
GET /api/recommendations/{user_id}
```

Parametreler:
- `limit`: Döndürülecek öneri sayısı (varsayılan: 10)
- `content_type`: "movie" veya "tv" (varsayılan: her ikisi)
- `strategy`: "collaborative", "content_based", "hybrid" (varsayılan: "hybrid")

### Model Eğitimi

```
POST /api/train
```

Header:
- `Authorization`: Bearer token

Body:
```json
{
  "model_type": "hybrid", 
  "params": {
    "learning_rate": 0.01,
    "n_factors": 100
  }
}
```

## Test

```bash
pytest tests/
```

## Lisans

Bu proje özel kullanım içindir ve tüm hakları saklıdır. 