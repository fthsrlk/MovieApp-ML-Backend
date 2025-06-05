# 🤖 MovieApp ML Backend

Modern hibrit film öneri sistemi - Python Flask backend, makine öğrenmesi algoritmaları ve TMDb API entegrasyonu ile gelişmiş film önerileri.

## ✨ Özellikler

### 🧠 Makine Öğrenmesi Özellikleri
- **Hibrit Öneri Sistemi**: Collaborative + Content-based filtering
- **Scikit-learn**: TF-IDF, Cosine Similarity, KNN algoritmaları
- **Real-time Learning**: Kullanıcı etkileşimlerinden öğrenme
- **Çoklu Algoritma Desteği**: Farklı senaryolar için optimize edilmiş modeller

### 🌐 Web API
- **Flask RESTful API**: Modern ve scalable backend architecture
- **JWT Authentication**: Güvenli kullanıcı kimlik doğrulama
- **Rate Limiting**: API abuse koruması
- **CORS Support**: Cross-origin istekleri desteği

### 🎯 Akıllı Öneri Motorları

#### Content-Based Filtering
- **Genre Similarity**: Tür benzerlikleri
- **Cast & Crew**: Oyuncu ve yönetmen benzerlikler
- **Plot Analysis**: Hikaye içeriği analizi
- **TF-IDF Vectorization**: Metin tabanlı benzerlik

#### Collaborative Filtering
- **User-User**: Benzer kullanıcı tercihleri
- **Item-Item**: Film/dizi benzerlikleri
- **Matrix Factorization**: Latent factor modelleri
- **Cold Start Problem**: Yeni kullanıcılar için çözümler

#### Hybrid Approach
- **Weighted Combination**: Algoritma ağırlıklandırması
- **Switch Hybrid**: Duruma göre algoritma seçimi
- **Feature Combination**: Çoklu özellik entegrasyonu

### 📊 Veri İşleme
- **TMDb API Integration**: 500,000+ film ve dizi verisi
- **Data Preprocessing**: Temizlik ve normalizasyon
- **Feature Engineering**: Akıllı özellik çıkarımı
- **Real-time Updates**: Canlı veri senkronizasyonu

## 🏗️ Teknik Mimari

### 🔧 Teknoloji Stack
```python
Backend Framework: Flask 2.3+
ML Libraries: scikit-learn, pandas, numpy
Database: SQLite (production'da PostgreSQL)
Caching: Redis (optional)
API: TMDb API v3
Authentication: JWT tokens
Deployment: Docker, Gunicorn, Nginx
```

### 📦 Proje Yapısı
```
ml_recommendation_engine/
├── api/                    # Flask API endpoints
│   ├── auth.py            # Authentication routes
│   ├── movies.py          # Film endpoints
│   ├── recommendations.py # Öneri endpoints
│   └── users.py           # Kullanıcı endpoints
├── app/                   # Web application
│   ├── templates/         # HTML templates
│   ├── static/           # CSS, JS, images
│   └── forms.py          # WTForms
├── data/                  # Veri dosyaları
│   ├── loader.py         # Veri yükleme
│   ├── preprocessor.py   # Veri işleme
│   └── movies.csv        # Film veri seti
├── models/               # ML modelleri
│   ├── collaborative.py # Collaborative filtering
│   ├── content_based.py  # Content-based filtering
│   ├── hybrid.py         # Hibrit sistem
│   └── trainer.py        # Model eğitimi
├── main.py              # Ana uygulama
├── config.py            # Konfigürasyon
└── requirements.txt     # Python dependencies
```

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.8+
- pip (Python package manager)
- TMDb API Key
- (Opsiyonel) Redis server

### Hızlı Başlangıç

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/fthsrlk/MovieApp-ML-Backend.git
cd MovieApp-ML-Backend
```

2. **Virtual environment oluşturun:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

3. **Dependencies yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Environment variables ayarlayın:**
```bash
cp .env.example .env
# .env dosyasını düzenleyin:
TMDB_API_KEY=your_api_key_here
SECRET_KEY=your_secret_key
FLASK_ENV=development
```

5. **Veritabanını başlatın:**
```bash
python -c "from main import create_app; create_app().app_context().push(); from models import db; db.create_all()"
```

6. **Uygulamayı çalıştırın:**
```bash
python main.py
```

### Docker ile Çalıştırma

```bash
# Docker image oluştur
docker build -t movieapp-ml .

# Container çalıştır
docker run -p 5000:5000 -e TMDB_API_KEY=your_key movieapp-ml
```

## 📡 API Endpoints

### Authentication
```http
POST /api/auth/register    # Kullanıcı kaydı
POST /api/auth/login       # Giriş yapma
POST /api/auth/logout      # Çıkış yapma
GET  /api/auth/profile     # Profil bilgileri
```

### Movies & TV Shows
```http
GET  /api/movies/search    # Film arama
GET  /api/movies/{id}      # Film detayları
GET  /api/movies/popular   # Popüler filmler
GET  /api/movies/trending  # Trend filmler
POST /api/movies/{id}/rate # Film puanlama
```

### Recommendations
```http
GET  /api/recommendations/movies/{user_id}     # Kişiselleştirilmiş öneriler
GET  /api/recommendations/similar/{movie_id}   # Benzer filmler
GET  /api/recommendations/trending             # Trend öneriler
POST /api/recommendations/feedback             # Öneri geri bildirimi
```

## 🧮 Makine Öğrenmesi Algoritmaları

### Content-Based Filtering
```python
# TF-IDF ile içerik benzerliği
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Film özelliklerini vektörize et
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_features)

# Cosine similarity hesapla
similarity_matrix = cosine_similarity(tfidf_matrix)
```

### Collaborative Filtering
```python
# KNN ile collaborative filtering
from sklearn.neighbors import NearestNeighbors

# User-item matrix oluştur
user_item_matrix = ratings.pivot_table(
    index='user_id', columns='movie_id', values='rating'
)

# KNN modeli eğit
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix.fillna(0))
```

### Hybrid System
```python
# Hibrit sistem - ağırlıklı kombinasyon
def hybrid_recommendation(user_id, movie_id, weights=[0.6, 0.4]):
    content_score = content_based_similarity(movie_id)
    collab_score = collaborative_similarity(user_id, movie_id)
    
    return weights[0] * content_score + weights[1] * collab_score
```

## 📊 Performans Metrikleri

### Model Değerlendirme
- **Precision@K**: Top-K önerilerin kesinliği
- **Recall@K**: Top-K önerilerin duyarlılığı
- **F1-Score**: Harmonic mean of precision and recall
- **RMSE**: Root Mean Square Error for rating prediction
- **Coverage**: Sistem tarafından önerilen unique items

### Benchmarks
```
Content-Based Model:
- Precision@10: 0.78
- Recall@10: 0.65
- Coverage: 0.92

Collaborative Model:
- Precision@10: 0.82
- Recall@10: 0.71
- RMSE: 0.89

Hybrid Model:
- Precision@10: 0.85
- Recall@10: 0.74
- F1-Score: 0.79
```

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-ml-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing ML feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-ml-feature`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📞 İletişim

**Fatih Şarlak**
- GitHub: [@fthsrlk](https://github.com/fthsrlk)
- Email: [email@example.com]

## 🙏 Teşekkürler

- [TMDb](https://www.themoviedb.org/) - Film verileri için
- [scikit-learn](https://scikit-learn.org/) - ML kütüphanesi için
- [Flask](https://flask.palletsprojects.com/) - Web framework için
- [MovieLens](https://grouplens.org/datasets/movielens/) - Araştırma veri seti için

---

⭐ **Bu projeyi beğendiyseniz star vermeyi unutmayın!**