import os
import json
import requests
import pickle
import pandas as pd
from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_cors import CORS
import logging
from dotenv import load_dotenv
import sys
import time
import datetime

# Mevcut dizini ve ml_recommendation_engine'i ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'ml_recommendation_engine'))

# ML Recommendation Engine modüllerini içe aktar
try:
    from ml_recommendation_engine.models.collaborative import CollaborativeFiltering
    from ml_recommendation_engine.models.content_based import ContentBasedFiltering
    from ml_recommendation_engine.models.hybrid import HybridRecommender
except ImportError as e:
    print(f"Import hatası: {e}")
    print("ml_recommendation_engine modülleri yüklenemedi!")

# Ortam değişkenlerini yükle
load_dotenv()

# Flask uygulamasını başlat
app = Flask(__name__)

# Jinja environment'ına strftime filtresini ekle
# app.jinja_env.filters['strftime'] = datetime.datetime.strftime

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.datetime.now().year}

@app.context_processor
def inject_user_watchlist():
    global user_watchlist # Global user_watchlist değişkenine erişim
    return {'user_watchlist': user_watchlist}

CORS(app)  # CORS desteği ekle

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Yapılandırma
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '3bc6092717beedc3a05e4d0809435ef2')
MODEL_DIR = os.getenv('MODEL_DIR', 'ml_recommendation_engine/models')
DATA_DIR = os.getenv('DATA_DIR', 'ml_recommendation_engine/data')

# Model ve veri yolları
CF_MODEL_PATH = os.path.join(MODEL_DIR, 'collaborative_model.pkl')
CB_MODEL_PATH = os.path.join(MODEL_DIR, 'content_based_model.pkl')
HYBRID_MODEL_PATH = os.path.join(MODEL_DIR, 'hybrid_model.pkl')
ITEMS_DATA_PATH = os.path.join(DATA_DIR, 'items.csv')
RATINGS_DATA_PATH = os.path.join(DATA_DIR, 'ratings.csv')

# Dizinleri oluştur
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Global değişkenler
cf_model = None
cb_model = None
hybrid_model = None
items_df = None
ratings_df = None
user_watchlist = {}  # Kullanıcının izleme listesi (tek kullanıcı olduğu için basit bir sözlük)

# --- Yardımcı Fonksiyonlar ---
def safe_parse_json(value):
    """JSON veya dict string'i güvenli bir şekilde parse et."""
    if pd.isna(value) or value is None:
        return {}
    if isinstance(value, dict):
        return value
    
    # Önce JSON olarak parse etmeyi dene
    try:
        return json.loads(value) 
    except:
        # Sonra Python dict stringi olarak dene
        try:
            import ast
            return ast.literal_eval(value)
        except:
            # Son çare
            return {}

def is_movie_id(item_id):
    """ID'nin film mi dizi mi olduğunu belirle. Birçok kaynağa göre, TMDB'de filmler genellikle
    küçük ID'lere (100000'den küçük), diziler ise daha büyük ID'lere sahiptir."""
    try:
        item_id = int(item_id)
        return item_id < 100000
    except:
        # Sayısal olmayan ID'ler için varsayılan film
        return True

# --- Yardımcı: CB modeli yeniden inşa et ---
def rebuild_content_based_model():
    """items_df güncel haline göre içerik tabanlı modeli sıfırdan oluşturur."""
    global cb_model, items_df
    try:
        if items_df is None or items_df.empty:
            return
        logger.info(f"İçerik tabanlı model oluşturuluyor. Mevcut {len(items_df)} öğe var.")
        cb_model = ContentBasedFiltering()
        cb_model.fit(items_df)
        # Diskte de sakla (isteğe bağlı)
        try:
            os.makedirs(os.path.dirname(CB_MODEL_PATH), exist_ok=True)
            cb_model.save(CB_MODEL_PATH)
        except Exception:
            pass
        logger.info("İçerik tabanlı model yeniden oluşturuldu.")
    except Exception as e:
        logger.error(f"İçerik tabanlı model oluşturulurken hata: {str(e)}")

# --- Yardımcı: items_df'e yeni içerik ekle ---
def add_item_metadata(item_details, media_type='movie'):
    """TMDB detaylarını items_df'e ekler (zaten varsa atlar) ve modeli günceller."""
    global items_df
    if not item_details or 'id' not in item_details:
        return
    item_id = item_details['id']
    if items_df is not None and not items_df.empty and (items_df['id'] == item_id).any():
        return  # zaten var

    title = item_details.get('title') or item_details.get('name', 'Bilinmeyen')
    overview = item_details.get('overview', '')
    release_date = item_details.get('release_date') or item_details.get('first_air_date', '')
    poster_path = item_details.get('poster_path', '')

    new_item = {
        'id': item_id,
        'title': title,
        'overview': overview,
        'release_date': release_date,
        'content_type': media_type,
        'poster_path': poster_path,
        'tmdb_details': item_details
    }
    if items_df is None or items_df.empty:
        items_df = pd.DataFrame([new_item])
    else:
        items_df = pd.concat([items_df, pd.DataFrame([new_item])], ignore_index=True)

    # Kaydet
    try:
        os.makedirs(os.path.dirname(ITEMS_DATA_PATH), exist_ok=True)
        items_df.to_csv(ITEMS_DATA_PATH, index=False)
    except Exception:
        pass

    # CB model güncelle
    rebuild_content_based_model()

# --- WATCHLIST TABANLI ÖNERİ ---
# Kullanıcının izleme listesine (rating gerektirmeden) dayalı içerik tabanlı öneriler üretir
# Mevcut ContentBasedFiltering modelinin benzerlik fonksiyonunu kullanır
def recommend_from_watchlist(cb_model, watchlist_ids, n=10, exclude_ids=None):
    """
    İzleme listesine benzeyen içerikleri döndür.
    
    Args:
        cb_model (ContentBasedFiltering): Eğitilmiş içerik tabanlı model
        watchlist_ids (list[int]): Kullanıcının izleme listesindeki item_id'ler
        n (int): Öneri sayısı
        exclude_ids (set[int], optional): Hariç tutulacak item_id'ler
    
    Returns:
        list[tuple]: (item_id, score) şeklinde öneriler
    """
    if cb_model is None or not watchlist_ids:
        logger.info("CB model yok veya izleme listesi boş, öneri yapılamıyor")
        return []
    
    # Logla watchlist içeriğini
    logger.info(f"İzleme listesi: {watchlist_ids}")
    
    if exclude_ids is None:
        exclude_ids = set()
    else:
        exclude_ids = set(exclude_ids)
    
    # İzleme listesindeki öğeleri de hariç tut
    exclude_ids.update(watchlist_ids)
    
    # Önce modeli yeniden oluştur - temiz başlangıç için
    # Bu adım maliyetli ama önerilerin doğruluğunu garanti eder
    from ml_recommendation_engine.models.content_based import ContentBasedFiltering
    temp_model = ContentBasedFiltering()
    if hasattr(cb_model, 'items_df') and cb_model.items_df is not None and not cb_model.items_df.empty:
        logger.info(f"Modeli izleme listesi için taze verilerle yeniden hesaplıyorum... ({len(cb_model.items_df)} öğe)")
        # Modeli taze veri ile eğit
        temp_model.fit(cb_model.items_df)
    else:
        logger.warning("CB modelin items_df özelliği yok veya boş - model verisi eksik olabilir")
        global items_df
        if items_df is not None and not items_df.empty:
            logger.info(f"Global items_df kullanılıyor: {len(items_df)} öğe")
            temp_model.fit(items_df)
        else:
            logger.error("Öneri için kullanılabilecek veri bulunamadı")
            return []
    
    # İçerik tabanlı öneriler için
    aggregate_scores = {}
    
    # Her izleme listesi öğesi için benzer öğeleri bul
    for watchlist_id in watchlist_ids:
        try:
            # Yeni modelimizde öğenin indeksini bul
            item_idx = temp_model.item_indices.get(watchlist_id)
            if item_idx is None:
                logger.warning(f"ID: {watchlist_id} modelimizde bulunamadı, atlanıyor")
                continue
                
            # Modelde olan öğeler için benzerlik skoru hesapla
            # En yüksek benzerlik skoruna sahip öğeleri al (daha fazla aday)
            similar_items = temp_model.get_similar_items(watchlist_id, n=max(100, n*5))
            
            if not similar_items:
                logger.warning(f"ID: {watchlist_id} için benzer öğe bulunamadı")
                continue
                
            # Her benzer öğe için skor topla (aynı öğe birden fazla kez önerilirse skoru artar)
            for item_id, score in similar_items:
                if item_id not in exclude_ids:
                    # Bu öğe zaten aggregate_scores'da varsa, skorunu güncelle
                    if item_id in aggregate_scores:
                        # Eğer öğe birden fazla izleme listesi öğesiyle eşleşiyorsa
                        # skorunu artır (daha yüksek ağırlık ver)
                        aggregate_scores[item_id] = aggregate_scores[item_id] + score
                    else:
                        aggregate_scores[item_id] = score
        except Exception as e:
            logger.error(f"Benzer öğeler hesaplanırken hata: {str(e)}")
    
    # Skorlara göre sırala ve en yüksek puanlı n tane öğeyi döndür
    sorted_items = sorted(aggregate_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Eğer ML modelinden yeterli öneri bulunamadıysa, TMDB API'den benzer içerikleri çek
    if len(sorted_items) < n / 2:
        logger.info(f"Yeterli öneri bulunamadı (sadece {len(sorted_items)} öğe), TMDB API'ye başvuruluyor")
        
        # Yeni öneriler listesi
        tmdb_recommendations = []
        
        # Her izleme listesi öğesi için TMDB'den benzer içerikleri çek
        for watchlist_id in watchlist_ids:
            # İçerik türünü belirle (film veya dizi)
            if is_movie_id(watchlist_id):
                media_type = 'movie'
            else:
                media_type = 'tv'
                
            # TMDB API'den benzer içerikleri çek
            try:
                similar_response = get_recommendations_tmdb(watchlist_id, media_type)
                similar_items = similar_response.get('results', [])
                
                # İzleme listesindeki ve exclude_ids'deki öğeleri çıkar
                similar_items = [item for item in similar_items if item['id'] not in exclude_ids]
                
                # Her benzer içerik için
                for idx, similar_item in enumerate(similar_items[:10]):
                    similar_id = similar_item.get('id')
                    
                    # Eğer bu öğe zaten recommendations içinde varsa atla
                    if any(rec_id == similar_id for rec_id, _ in sorted_items) or \
                       any(rec_id == similar_id for rec_id, _ in tmdb_recommendations):
                        continue
                    
                    # 1.0'dan 0.0'a doğru azalan yapay skor hesapla (sıralamada daha düşük olsun)
                    artificial_score = 1.0 - (idx * 0.1)
                    
                    # Öneriler listesine ekle
                    tmdb_recommendations.append((similar_id, artificial_score))
                    
                    # Her benzer içeriğin detaylarını çek ve items_df'e ekle
                    if media_type == 'movie':
                        similar_details = get_movie_details(similar_id)
                    else:
                        similar_details = get_tv_details(similar_id)
                    
                    if similar_details:
                        add_item_metadata(similar_details, media_type)
            except Exception as e:
                logger.error(f"TMDB benzer içerikleri çekerken hata: {str(e)}")
        
        # TMDB önerilerini ML önerilerine ekle
        sorted_items.extend(tmdb_recommendations)
        # Tekrar sırala
        sorted_items = sorted(sorted_items, key=lambda x: x[1], reverse=True)
    
    if not sorted_items:
        logger.warning("Hiçbir benzer öğe bulunamadı")
        return []
    
    # İzleme listesi içeriğindeki tür/özellik dağılımı analiz et
    top_n = sorted_items[:n]
    logger.info(f"Bulunan benzer içerikler: {top_n}")
    
    return top_n

# TMDB API için yardımcı fonksiyonlar
def search_movies(query, page=1):
    """
    TMDB API ile film araması yapar
    """
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": query,
        "language": "tr-TR",
        "page": page,
        "include_adult": False
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"TMDB API hata: {response.status_code}")
            return {"results": []}
    except requests.exceptions.RequestException as e:
        logger.error(f"Film arama sırasında bağlantı hatası: {str(e)}")
        return {"results": [], "error": "İnternet bağlantısı sorunu. Lütfen bağlantınızı kontrol edin."}
    except Exception as e:
        logger.error(f"Film arama sırasında beklenmeyen hata: {str(e)}")
        return {"results": [], "error": "Bir hata oluştu."}

def search_tv_series(query, page=1):
    """
    TMDB API ile dizi araması yapar
    """
    url = f"https://api.themoviedb.org/3/search/tv"
    params = {
        "api_key": TMDB_API_KEY,
        "query": query,
        "language": "tr-TR",
        "page": page,
        "include_adult": False
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"TMDB API hata: {response.status_code}")
            return {"results": []}
    except requests.exceptions.RequestException as e:
        logger.error(f"Dizi arama sırasında bağlantı hatası: {str(e)}")
        return {"results": [], "error": "İnternet bağlantısı sorunu. Lütfen bağlantınızı kontrol edin."}
    except Exception as e:
        logger.error(f"Dizi arama sırasında beklenmeyen hata: {str(e)}")
        return {"results": [], "error": "Bir hata oluştu."}

def get_movie_details(movie_id):
    """
    TMDB API'den film detaylarını alır
    """
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=tr-TR&append_to_response=credits,keywords"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Film detayları alınırken hata: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Film detayları alınırken hata: {str(e)}")
        return None

def get_tv_details(tv_id):
    """
    TMDB API'den dizi detaylarını alır
    """
    try:
        url = f"https://api.themoviedb.org/3/tv/{tv_id}?api_key={TMDB_API_KEY}&language=tr-TR&append_to_response=credits,keywords"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Dizi detayları alınırken hata: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Dizi detayları alınırken hata: {str(e)}")
        return None

def get_recommendations_tmdb(item_id, media_type='movie'):
    """
    TMDB API ile öneriler alır
    """
    url = f"https://api.themoviedb.org/3/{media_type}/{item_id}/recommendations"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "tr-TR",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return {"results": []}

# Model yönetimi fonksiyonları
def load_models():
    """
    ML modellerini yükler
    """
    global cf_model, cb_model, hybrid_model
    
    try:
        # ML modülünü import et
        from ml_recommendation_engine.models.collaborative import CollaborativeFiltering
        from ml_recommendation_engine.models.content_based import ContentBasedFiltering
        from ml_recommendation_engine.models.hybrid import HybridRecommender
        
        # Hibrit model
        if os.path.exists(HYBRID_MODEL_PATH):
            hybrid_model = HybridRecommender.load(HYBRID_MODEL_PATH)
            logger.info("Hibrit model yüklendi")
        else:
            logger.warning(f"Hibrit model bulunamadı: {HYBRID_MODEL_PATH}")
        
        # İşbirlikçi filtreleme modeli
        if os.path.exists(CF_MODEL_PATH):
            cf_model = CollaborativeFiltering.load(CF_MODEL_PATH)
            logger.info("İşbirlikçi filtreleme modeli yüklendi")
        else:
            logger.warning(f"İşbirlikçi filtreleme modeli bulunamadı: {CF_MODEL_PATH}")
        
        # İçerik tabanlı filtreleme modeli
        if os.path.exists(CB_MODEL_PATH):
            cb_model = ContentBasedFiltering.load(CB_MODEL_PATH)
            logger.info("İçerik tabanlı filtreleme modeli yüklendi")
        else:
            logger.warning(f"İçerik tabanlı filtreleme modeli bulunamadı: {CB_MODEL_PATH}")
        
        return True
    except Exception as e:
        logger.error(f"Model yükleme hatası: {str(e)}")
        logger.error(f"Hata detayı: {str(e.__traceback__)}")
        return False

def load_data():
    """
    Veri dosyalarını yükler
    """
    global items_df, ratings_df
    
    try:
        # items_df yükle (varsa)
        if os.path.exists(ITEMS_DATA_PATH):
            items_df = pd.read_csv(ITEMS_DATA_PATH)
            # tmdb_details'ı JSON'a dönüştür
            # (not: boş, na veya null değerler problemli olmamalı)
            if 'tmdb_details' in items_df.columns:
                items_df['tmdb_details'] = items_df['tmdb_details'].apply(
                    lambda x: safe_parse_json(x) if pd.notna(x) else {}
                )
            logger.info(f"{len(items_df)} öğe yüklendi")
        else:
            logger.warning(f"{ITEMS_DATA_PATH} bulunamadı, boş DataFrame oluşturuldu")
            items_df = pd.DataFrame()
            
        # ÖNEMLI: Sütun adı tutarlılığı
        # item_id -> id olarak değiştirelim (her ikisini de kabul edelim)
        if 'item_id' in items_df.columns and 'id' not in items_df.columns:
            items_df = items_df.rename(columns={'item_id': 'id'})
        
        # ratings_df yükle (varsa)
        if os.path.exists(RATINGS_DATA_PATH):
            ratings_df = pd.read_csv(RATINGS_DATA_PATH)
            logger.info(f"{len(ratings_df)} değerlendirme yüklendi")
        else:
            logger.warning(f"{RATINGS_DATA_PATH} bulunamadı, boş DataFrame oluşturuldu")
            ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
            
    except Exception as e:
        logger.error(f"Veriler yüklenirken hata: {str(e)}")
        items_df = pd.DataFrame()
        ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])

def add_rating(user_id, item_id, rating, item_details=None):
    """
    Kullanıcı değerlendirmesi ekler
    """
    global ratings_df, items_df, cb_model
    
    try:
        # ratings_df ve items_df'in en güncel halini yükle
        load_data()

        # Mevcut değerlendirmeyi kontrol et (ratings_df boş olsa bile çalışır)
        existing_rating = pd.DataFrame()
        if ratings_df is not None and not ratings_df.empty:
            existing_rating = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['item_id'] == item_id)]

        if not existing_rating.empty:
            # Mevcut değerlendirmeyi güncelle
            ratings_df.loc[(ratings_df['user_id'] == user_id) & (ratings_df['item_id'] == item_id), 'rating'] = float(rating)
            logger.info(f"Değerlendirme güncellendi: Kullanıcı {user_id}, İçerik {item_id}, Puan {rating}")
        else:
            # Yeni değerlendirme ekle
            import time
            new_rating = pd.DataFrame({
                'user_id': [user_id],
                'item_id': [item_id],
                'rating': [float(rating)],
                'timestamp': [int(time.time())]
            })
            if ratings_df is None or ratings_df.empty:
                ratings_df = new_rating
            else:
                ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
            logger.info(f"Yeni değerlendirme eklendi: Kullanıcı {user_id}, İçerik {item_id}, Puan {rating}")
        
        # items_df yoksa oluştur
        if items_df is None:
            items_df = pd.DataFrame(columns=['id', 'title', 'overview', 'release_date', 'content_type', 'poster_path'])
        
        # İçerik bilgilerini items_df'e ekle (yoksa)
        if item_details and items_df is not None:
            existing_item = items_df[items_df['id'] == item_id]
            if existing_item.empty:
                content_type = item_details.get('media_type', 'movie')
                title = item_details.get('title', item_details.get('name', 'Bilinmeyen'))
                overview = item_details.get('overview', '')
                release_date = item_details.get('release_date', item_details.get('first_air_date', ''))
                poster_path = item_details.get('poster_path', '')
                
                new_item = pd.DataFrame({
                    'id': [item_id],
                    'title': [title],
                    'overview': [overview],
                    'release_date': [release_date],
                    'content_type': [content_type],
                    'poster_path': [poster_path]
                })
                items_df = pd.concat([items_df, new_item], ignore_index=True)
                logger.info(f"Yeni içerik eklendi: {item_id} - {title}")
        
        # Veri kaydetme işlemi
        try:
            os.makedirs(os.path.dirname(RATINGS_DATA_PATH), exist_ok=True)
            ratings_df.to_csv(RATINGS_DATA_PATH, index=False)
            
            if items_df is not None:
                os.makedirs(os.path.dirname(ITEMS_DATA_PATH), exist_ok=True)
                items_df.to_csv(ITEMS_DATA_PATH, index=False)
            
            logger.info("Değerlendirme verileri başarıyla kaydedildi")

            # İçerik tabanlı modelin kullanıcı profilini güncelle (eğer model yüklü ise)
            if cb_model and ratings_df is not None:
                logger.info(f"Kullanıcı {user_id} için CB model profili güncelleniyor.")
                # ratings_df'in güncel kopyasını kullan
                current_user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()
                if not current_user_ratings.empty:
                    # _create_user_profile_for_single_user metodu user_ratings_df bekliyor,
                    # bu nedenle tüm ratings_df'i değil, sadece ilgili kullanıcınınkini verelim.
                    # Ancak metodun orijinal implementasyonu tüm ratings_df'i alıp içinden filtreliyor olabilir,
                    # bu yüzden en iyisi metodun kendisine tüm ratings_df'i vermek ve onun filtrelemesine izin vermek.
                    # Emin olmak için ContentBasedFiltering._create_user_profile_for_single_user tanımını kontrol ettim.
                    # Metot, tüm user_ratings_df'i değil, sadece ilgili kullanıcıya ait olanları bekliyor.
                    # Ancak, fit metodu içindeki _create_user_profiles tüm ratings_df'i alıyor.
                    # Hibrit modeldeki dinamik profil oluşturma da tüm ratings_df'i alıyor.
                    # Şimdilik, en son değerlendirmelerle tüm ratings_df'i cb_model'e verelim.
                    # cb_model._create_user_profiles(ratings_df) # Bu tüm kullanıcıları günceller, yavaş olabilir.
                    # Sadece mevcut kullanıcıyı güncelleyelim:
                    if hasattr(cb_model, '_create_user_profile_for_single_user'):
                         # ratings_df'in en güncel halini kullandığından emin olalım.
                        cb_model._create_user_profile_for_single_user(user_id, ratings_df[ratings_df['user_id'] == user_id])
                        logger.info(f"Kullanıcı {user_id} için CB profili güncellendi.")
                    else:
                        logger.warning("cb_model'de _create_user_profile_for_single_user metodu bulunamadı.")
                else:
                    logger.info(f"Kullanıcı {user_id} için güncellenecek değerlendirme bulunamadı (CB profili).")
            
            return True
        except Exception as e:
            logger.error(f"Veri kaydetme veya CB profil güncelleme hatası: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Değerlendirme ekleme hatası: {str(e)}")
        return False

def get_ml_recommendations(user_id, n=10, strategy='watchlist'):
    """
    Kullanıcıya makine öğrenimi modelleriyle önerileri döndürür
    """
    global items_df, cb_model
    
    # Veri dosyalarını güncelle
    load_data()

    # Sadece içerik tabanlı model yeterli, yoksa oluştur
    if cb_model is None:
        rebuild_content_based_model()
    
    if cb_model is None:
        logger.error("İçerik tabanlı model yüklenemedi, öneri oluşturulamıyor.")
        return []
    
    try:
        # İzleme listesini kontrol et
        movie_ids = user_watchlist.get('movie', [])
        tv_ids = user_watchlist.get('tv', [])
        
        watchlist_ids = movie_ids + tv_ids
        
        if not watchlist_ids:
            logger.info(f"Kullanıcı {user_id} izleme listesi boş, öneri yapılamıyor")
            return []
        
        logger.info(f"İzleme listesindeki öğeler: {watchlist_ids}")
        
        # İzleme listesinden öneriler
        logger.info("İzleme listesine dayalı öneriler oluşturuluyor")
        recs = recommend_from_watchlist(cb_model, watchlist_ids, n=n)
        
        if not recs:
            logger.warning("İzleme listesinden öneri oluşturulamadı")
            return []
        
        logger.info(f"Oluşturulan öneriler: {recs}")
        
        # Önerileri zenginleştir (TMDB API'den detaylar)
        enriched_recs = []
        for item_id, score in recs:
            # Önce veritabanımızda bu öğe var mı kontrol et
            found_in_db = False
            if items_df is not None and not items_df.empty:
                item_df = items_df[items_df['id'] == item_id]
                if not item_df.empty:
                    item = item_df.iloc[0].to_dict()
                    # Her durumda eksik detayları tamamla
                    if not item.get('poster_path') or not item.get('overview'):
                        try:
                            if is_movie_id(item_id):
                                details = get_movie_details(item_id)
                                if details:
                                    item['poster_path'] = details.get('poster_path')
                                    item['overview'] = details.get('overview')
                                    item['title'] = details.get('title', item.get('title'))
                                    item['tmdb_details'] = details
                            else:
                                details = get_tv_details(item_id)
                                if details:
                                    item['poster_path'] = details.get('poster_path')
                                    item['overview'] = details.get('overview')
                                    item['title'] = details.get('name', item.get('title'))
                                    item['tmdb_details'] = details
                        except Exception as e:
                            logger.warning(f"Öğe {item_id} detayları güncellenirken hata: {str(e)}")
                    
                    # Skoru ekle ve listeye ekle
                    item['score'] = score
                    item['item_id'] = item_id  # Eski template için geriye dönük uyumluluk
                    enriched_recs.append({'item_id': item_id, 'score': score, 'info': item})
                    found_in_db = True
            
            # Veritabanında yoksa TMDB'den getir
            if not found_in_db:
                try:
                    if is_movie_id(item_id):
                        details = get_movie_details(item_id)
                        if details:
                            # Template'in beklediği formata dönüştür
                            enriched_recs.append({
                                'item_id': item_id,
                                'score': score,
                                'info': {
                                    'id': item_id,
                                    'title': details.get('title', ''),
                                    'overview': details.get('overview', ''),
                                    'poster_path': details.get('poster_path', ''),
                                    'content_type': 'movie',
                                    'release_date': details.get('release_date', ''),
                                    'tmdb_details': details
                                }
                            })
                    else:
                        details = get_tv_details(item_id)
                        if details:
                            # Template'in beklediği formata dönüştür
                            enriched_recs.append({
                                'item_id': item_id,
                                'score': score,
                                'info': {
                                    'id': item_id,
                                    'title': details.get('name', ''),
                                    'overview': details.get('overview', ''),
                                    'poster_path': details.get('poster_path', ''),
                                    'content_type': 'tv',
                                    'release_date': details.get('first_air_date', ''),
                                    'tmdb_details': details
                                }
                            })
                except Exception as e:
                    logger.error(f"Öğe {item_id} detayları alınırken hata: {str(e)}")
        
        logger.info(f"{len(enriched_recs)} adet öneri bulundu.")
        return enriched_recs
        
    except Exception as e:
        logger.error(f"Öneriler oluşturulurken hata: {str(e)}")
        return []

# Flask rotaları
@app.route('/')
def index():
    """
    Ana sayfa
    """
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    """
    Arama sayfası
    """
    query = request.args.get('q', '')
    
    movie_results = []
    tv_results = []
    error_message = None
    
    if query:
        # Hem film hem dizi araması yap
        movie_response = search_movies(query)
        tv_response = search_tv_series(query)
        
        movie_results = movie_response.get('results', [])
        tv_results = tv_response.get('results', [])
        
        # API bağlantı hatası kontrolü
        if 'error' in movie_response or 'error' in tv_response:
            error_message = movie_response.get('error') or tv_response.get('error')
            logger.warning(f"Arama sırasında hata: {error_message}")
    
    return render_template('search.html', 
                         query=query, 
                         movie_results=movie_results,
                         tv_results=tv_results,
                         error_message=error_message)

@app.route('/detail/<media_type>/<int:item_id>')
def detail(media_type, item_id):
    """
    Detay sayfası
    """
    if media_type == 'movie':
        item = get_movie_details(item_id)
        title = item.get('title', 'Bilinmeyen Film')
    else:
        item = get_tv_details(item_id)
        title = item.get('name', 'Bilinmeyen Dizi')
    
    # İzleme listesinde mi kontrol et
    in_watchlist = item_id in user_watchlist.get(media_type, [])
    
    # Benzer içerikler
    similar = get_recommendations_tmdb(item_id, media_type).get('results', [])
    
    return render_template('detail.html', 
                         item=item, 
                         title=title, 
                         media_type=media_type,
                         in_watchlist=in_watchlist,
                         similar=similar)

@app.route('/watchlist')
def watchlist():
    """
    İzleme listesi sayfası
    """
    movie_items = []
    tv_items = []
    
    # Film listesi
    for movie_id in user_watchlist.get('movie', []):
        movie_details = get_movie_details(movie_id)
        if movie_details:
            movie_items.append(movie_details)
    
    # Dizi listesi
    for tv_id in user_watchlist.get('tv', []):
        tv_details = get_tv_details(tv_id)
        if tv_details:
            tv_items.append(tv_details)
    
    return render_template('watchlist.html', 
                         movie_items=movie_items, 
                         tv_items=tv_items)

@app.route('/recommendations')
def recommendations():
    """
    Öneriler sayfası
    """
    user_id = 1  # Tek kullanıcı olduğu için sabit
    # Her durumda içerik tabanlı strateji kullanılacak
    strategy = 'content_based'
    
    # İzleme listesi boş değilse ve content-based model de boşsa rebuild ederiz
    movie_ids = user_watchlist.get('movie', [])
    tv_ids = user_watchlist.get('tv', [])
    if (movie_ids or tv_ids) and cb_model is None:
        logger.info("İzleme listesi var ama CB model yok, yeniden oluşturuluyor")
        rebuild_content_based_model()
    
    # Kullanıcının izleme listesi doluysa ML önerileri al
    ml_recs = []
    if movie_ids or tv_ids:
        logger.info(f"Kullanıcı {user_id} için öneriler alınıyor")
        # Mevcut ratings_df'i log seviyesinde kontrol et
        if ratings_df is not None:
            logger.info(f"Ratings DataFrame'de {len(ratings_df)} kayıt var")
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            logger.info(f"Kullanıcı {user_id} için {len(user_ratings)} değerlendirme mevcut")
        
        # ML önerileri al
        ml_recs = get_ml_recommendations(user_id, n=20, strategy=strategy)
        logger.info(f"Toplam {len(ml_recs)} ML önerisi alındı")
    else:
        logger.info(f"Kullanıcı {user_id} için izleme listesi boş, ML önerileri alınamadı")
    
    return render_template('recommendations.html', 
                         ml_recommendations=ml_recs)

@app.route('/api/watchlist/add', methods=['POST'])
def add_to_watchlist():
    """
    İzleme listesine ekle API
    """
    global items_df
    data = request.json
    media_type = data.get('media_type', 'movie')
    item_id = data.get('item_id')
    
    if not item_id:
        return jsonify({'success': False, 'error': 'ID eksik'})
    
    # İzleme listesini başlat (yoksa)
    if media_type not in user_watchlist:
        user_watchlist[media_type] = []
    
    # Zaten listede mi kontrol et
    if item_id not in user_watchlist[media_type]:
        user_watchlist[media_type].append(item_id)
        
        # İçerik detaylarını al ve metadata'ya ekle
        if media_type == 'movie':
            item_details = get_movie_details(item_id)
        else:
            item_details = get_tv_details(item_id)
        
        if item_details:
            add_item_metadata(item_details, media_type)
            
            # TMDB'den benzer içerikleri çek ve veri setine ekle
            try:
                similar_response = get_recommendations_tmdb(item_id, media_type)
                similar_items = similar_response.get('results', [])
                
                logger.info(f"{len(similar_items)} benzer içerik TMDB'den çekildi")
                
                # İlk 10 benzer içeriği items_df'e ekle
                for idx, similar_item in enumerate(similar_items[:10]):
                    similar_id = similar_item.get('id')
                    # Her benzer içeriğin detaylarını çek
                    if media_type == 'movie':
                        similar_details = get_movie_details(similar_id)
                    else:
                        similar_details = get_tv_details(similar_id)
                    
                    if similar_details:
                        add_item_metadata(similar_details, media_type)
                
                # İçerik tabanlı modeli güncelle
                rebuild_content_based_model()
                
            except Exception as e:
                logger.error(f"Benzer içerikleri çekerken hata: {str(e)}")
    
    return jsonify({'success': True})

@app.route('/api/watchlist/remove', methods=['POST'])
def remove_from_watchlist():
    """
    İzleme listesinden çıkar API
    """
    data = request.json
    media_type = data.get('media_type', 'movie')
    item_id = data.get('item_id')
    
    if not item_id:
        return jsonify({'success': False, 'error': 'ID eksik'})
    
    # Listeden çıkar
    if media_type in user_watchlist and item_id in user_watchlist[media_type]:
        user_watchlist[media_type].remove(item_id)
    
    return jsonify({'success': True})

@app.route('/api/rate', methods=['POST'])
def rate_item():
    """
    İçerik değerlendirme API
    """
    data = request.json
    media_type = data.get('media_type', 'movie')
    item_id = data.get('item_id')
    rating = data.get('rating', 0)
    
    if not item_id:
        return jsonify({'success': False, 'error': 'ID eksik'})
    
    # İçerik detaylarını al
    if media_type == 'movie':
        item_details = get_movie_details(item_id)
    else:
        item_details = get_tv_details(item_id)
    
    # Değerlendirme ekle
    result = add_rating(1, item_id, rating, item_details)
    
    return jsonify({'success': result})

@app.route('/api/search', methods=['GET'])
def api_search():
    """
    Arama API - AJAX istekleri için
    """
    query = request.args.get('q', '')
    
    movie_results = []
    tv_results = []
    error = None
    
    if query:
        # Hem film hem dizi araması yap
        try:
            movie_response = search_movies(query)
            tv_response = search_tv_series(query)
            
            movie_results = movie_response.get('results', [])
            tv_results = tv_response.get('results', [])
            
            # API bağlantı hatası kontrolü
            if 'error' in movie_response or 'error' in tv_response:
                error = movie_response.get('error') or tv_response.get('error')
                logger.warning(f"AJAX arama sırasında hata: {error}")
        except Exception as e:
            error = "Arama sırasında bir hata oluştu"
            logger.error(f"AJAX arama hatası: {str(e)}")
    
    return jsonify({
        'movie_results': movie_results,
        'tv_results': tv_results,
        'error': error
    })

@app.route('/api/watchlist/get', methods=['GET'])
def get_watchlist():
    """
    İzleme listesini alma API
    """
    return jsonify({
        'success': True,
        'watchlist': user_watchlist
    })

@app.route('/api/clear_all_data', methods=['POST'])
def clear_all_data():
    """
    Tüm kullanıcı verilerini temizle
    """
    global user_watchlist, ratings_df, items_df
    
    try:
        # İzleme listesini temizle
        user_watchlist = {}
        
        # Değerlendirme verilerini temizle
        if ratings_df is not None:
            ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
            # Değişiklikleri kaydet
            os.makedirs(os.path.dirname(RATINGS_DATA_PATH), exist_ok=True)
            ratings_df.to_csv(RATINGS_DATA_PATH, index=False)
        
        # İzlenen içerikleri koruyabilir veya temizleyebiliriz
        # Burada temizlemeyi tercih ettik
        if items_df is not None:
            items_df = pd.DataFrame(columns=['id', 'title', 'overview', 'release_date', 'content_type', 'poster_path'])
            # Değişiklikleri kaydet
            os.makedirs(os.path.dirname(ITEMS_DATA_PATH), exist_ok=True)
            items_df.to_csv(ITEMS_DATA_PATH, index=False)
            
        logger.info("Tüm kullanıcı verileri başarıyla temizlendi")
        return jsonify({'success': True, 'message': 'Tüm veriler temizlendi'})
    except Exception as e:
        logger.error(f"Verileri temizleme hatası: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/hard_reset', methods=['GET', 'POST'])
def hard_reset():
    """Tüm verileri, modelleri ve izleme listesini sıfırlar, model dosyalarını da siler."""
    global items_df, ratings_df, cb_model, cf_model, hybrid_model, user_watchlist
    
    try:
        # İzleme listesini temizle
        user_watchlist = {}
        
        # Veri dosyalarını sıfırla
        if os.path.exists(ITEMS_DATA_PATH):
            os.remove(ITEMS_DATA_PATH)
            logger.info(f"{ITEMS_DATA_PATH} silindi")
            
        if os.path.exists(RATINGS_DATA_PATH):
            os.remove(RATINGS_DATA_PATH)
            logger.info(f"{RATINGS_DATA_PATH} silindi")
            
        # Modelleri sıfırla
        model_paths = [CB_MODEL_PATH, CF_MODEL_PATH, HYBRID_MODEL_PATH]
        for path in model_paths:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"{path} silindi")
        
        # Değişkenleri sıfırla
        items_df = pd.DataFrame()
        ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
        cb_model = None
        cf_model = None
        hybrid_model = None
        
        # Model klasörünü yeniden oluştur (yoksa)
        os.makedirs(os.path.dirname(CB_MODEL_PATH), exist_ok=True)
        
        logger.info("Tüm veriler ve modeller tamamen silindi")
        
        return jsonify({"success": True, "message": "Tüm veriler ve modeller tamamen silindi"})
    except Exception as e:
        logger.error(f"Hard reset sırasında hata: {str(e)}")
        return jsonify({"success": False, "message": f"Reset işlemi sırasında hata: {str(e)}"}), 500

# Reset endpoint
@app.route('/api/reset', methods=['GET', 'POST'])
def reset_app():
    """Tüm verileri, modelleri ve izleme listesini sıfırlar."""
    global items_df, ratings_df, cb_model, user_watchlist
    
    try:
        # İzleme listesini temizle
        user_watchlist = {}
        
        # Veri dosyalarını sıfırla
        if os.path.exists(ITEMS_DATA_PATH):
            os.remove(ITEMS_DATA_PATH)
            logger.info(f"{ITEMS_DATA_PATH} silindi")
            
        if os.path.exists(RATINGS_DATA_PATH):
            os.remove(RATINGS_DATA_PATH)
            logger.info(f"{RATINGS_DATA_PATH} silindi")
            
        # Modelleri sıfırla
        if os.path.exists(CB_MODEL_PATH):
            os.remove(CB_MODEL_PATH)
            logger.info(f"{CB_MODEL_PATH} silindi")
        
        # Yeni boş DataFrame'ler oluştur
        items_df = pd.DataFrame()
        ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
        cb_model = None
        
        # Popüler filmleri yükle - maksimum 100 film ile sınırla
        load_popular_movies(limit=50)
        load_popular_tv(limit=30)
        
        # ContentBased modeli yeniden oluştur
        rebuild_content_based_model()
        
        return jsonify({"success": True, "message": "Uygulama sıfırlandı, yeni veriler yüklendi"})
    except Exception as e:
        logger.error(f"Sıfırlama hatası: {str(e)}")
        return jsonify({"success": False, "message": f"Sıfırlama sırasında hata: {str(e)}"}), 500

# --- CLI komutu: Reset ve Popüler içerikleri yükle ---
if __name__ == '__main__':
    import sys
    
    # Komut satırı argümanlarını kontrol et
    if len(sys.argv) > 1 and sys.argv[1] == 'reset':
        logger.info("Tüm verileri sıfırlama ve popüler içerikleri yükleme işlemi başlatılıyor")
        
        # Eski veri dosyalarını sil
        if os.path.exists(ITEMS_DATA_PATH):
            os.remove(ITEMS_DATA_PATH)
            logger.info(f"{ITEMS_DATA_PATH} silindi")
            
        if os.path.exists(RATINGS_DATA_PATH):
            os.remove(RATINGS_DATA_PATH)
            logger.info(f"{RATINGS_DATA_PATH} silindi")
            
        # Eski model dosyalarını sil
        for model_path in [CF_MODEL_PATH, CB_MODEL_PATH, HYBRID_MODEL_PATH]:
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"{model_path} silindi")
        
        # TMDB'den popüler film ve dizileri yükle
        try:
            from ml_recommendation_engine.data.loader import TMDBDataLoader
            
            loader = TMDBDataLoader(api_key=TMDB_API_KEY)
            
            # Popüler film ve dizileri yükle (sadece metadata)
            logger.info("Popüler filmler ve diziler yükleniyor")
            
            # İlk 3 sayfa popüler film yükle
            movie_count = 0
            for page in range(1, 4):
                try:
                    movies = loader.get_popular_movies(page=page)
                    for movie in movies:
                        movie_id = movie['id']
                        movie_details = loader.get_movie_details(movie_id)
                        if movie_details:
                            add_item_metadata(movie_details, 'movie')
                            movie_count += 1
                            if movie_count % 10 == 0:
                                logger.info(f"{movie_count} film yüklendi")
                            time.sleep(0.1)  # Rate limit
                except Exception as e:
                    logger.error(f"Film yükleme hatası: {str(e)}")
            
            # İlk 2 sayfa popüler dizi yükle
            tv_count = 0
            for page in range(1, 3):
                try:
                    tv_shows = loader.get_popular_tv_shows(page=page)
                    for tv in tv_shows:
                        tv_id = tv['id']
                        tv_details = loader.get_tv_details(tv_id)
                        if tv_details:
                            add_item_metadata(tv_details, 'tv')
                            tv_count += 1
                            if tv_count % 10 == 0:
                                logger.info(f"{tv_count} dizi yüklendi")
                            time.sleep(0.1)  # Rate limit
                except Exception as e:
                    logger.error(f"Dizi yükleme hatası: {str(e)}")
            
            logger.info(f"Toplam {movie_count} film ve {tv_count} dizi yüklendi")
                
        except Exception as e:
            logger.error(f"İçerik yükleme hatası: {str(e)}")
        
        # CB modelini yeniden oluştur
        load_data()
        rebuild_content_based_model()
        
        logger.info("Sıfırlama işlemi tamamlandı")
        sys.exit(0)
    else:
        # Normal başlangıç: veri ve modelleri yükle
        load_data()
        load_models()
        
        # Flask uygulamasını yalnızca geliştirme ortamında başlat
        # Üretim ortamında (örn. Waitress ile) USE_PRODUCTION_SERVER=1 olarak ayarlandığında bu blok atlanır.
        if os.environ.get('USE_PRODUCTION_SERVER', '0') != '1':
            app.run(debug=True, host='0.0.0.0', port=5000)

def load_popular_movies(limit=50):
    """
    TMDB API'den popüler filmleri yükler ve items_df'e ekler
    """
    global items_df
    
    try:
        logger.info(f"Popüler filmler TMDB'den yükleniyor (max {limit} adet)")
        new_items = []
        
        # Her sayfada 20 film var, istenilen sayıya ulaşana kadar yükle
        page = 1
        max_page = (limit + 19) // 20  # Yukarı yuvarlama
        
        while len(new_items) < limit and page <= max_page:
            try:
                url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=tr-TR&page={page}"
                response = requests.get(url, timeout=5)
                
                if response.status_code != 200:
                    logger.error(f"Popüler filmler alınırken hata: {response.status_code}")
                    break
                    
                results = response.json().get('results', [])
                
                # Her film için detayları al ve yeni liste oluştur
                for movie in results:
                    if len(new_items) >= limit:
                        break
                        
                    movie_id = movie.get('id')
                    
                    # Film detaylarını al
                    details = get_movie_details(movie_id)
                    
                    if not details:
                        continue
                        
                    # Yeni öğe oluştur
                    new_item = {
                        'id': movie_id,
                        'title': details.get('title', ''),
                        'overview': details.get('overview', ''),
                        'release_date': details.get('release_date', ''),
                        'content_type': 'movie',
                        'poster_path': details.get('poster_path', ''),
                        'tmdb_details': json.dumps(details)
                    }
                    
                    new_items.append(new_item)
                    logger.info(f"Film yüklendi: {new_item['title']} (ID: {movie_id})")
                
                # Sonraki sayfa
                page += 1
                
                # API limiti aşılmaması için kısa bir duraklama
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Sayfa {page} yüklenirken hata: {str(e)}")
                break
                
        # Yeni DataFrame oluştur ve mevcut olanla birleştir
        if new_items:
            new_df = pd.DataFrame(new_items)
            
            if items_df is None or items_df.empty:
                items_df = new_df
            else:
                # Zaten var olan öğeleri çıkar
                existing_ids = set(items_df['id']) if 'id' in items_df.columns else set()
                new_df = new_df[~new_df['id'].isin(existing_ids)]
                
                # Birleştir
                items_df = pd.concat([items_df, new_df], ignore_index=True)
                
            # Dosyaya kaydet
            os.makedirs(os.path.dirname(ITEMS_DATA_PATH), exist_ok=True)
            items_df.to_csv(ITEMS_DATA_PATH, index=False)
            
            logger.info(f"{len(new_items)} film başarıyla yüklendi")
        else:
            logger.warning("Hiçbir film yüklenemedi")
            
    except Exception as e:
        logger.error(f"Popüler filmleri yükleme hatası: {str(e)}")
        
def load_popular_tv(limit=30):
    """
    TMDB API'den popüler dizileri yükler ve items_df'e ekler
    """
    global items_df
    
    try:
        logger.info(f"Popüler diziler TMDB'den yükleniyor (max {limit} adet)")
        new_items = []
        
        # Her sayfada 20 dizi var, istenilen sayıya ulaşana kadar yükle
        page = 1
        max_page = (limit + 19) // 20  # Yukarı yuvarlama
        
        while len(new_items) < limit and page <= max_page:
            try:
                url = f"https://api.themoviedb.org/3/tv/popular?api_key={TMDB_API_KEY}&language=tr-TR&page={page}"
                response = requests.get(url, timeout=5)
                
                if response.status_code != 200:
                    logger.error(f"Popüler diziler alınırken hata: {response.status_code}")
                    break
                    
                results = response.json().get('results', [])
                
                # Her dizi için detayları al ve yeni liste oluştur
                for tv in results:
                    if len(new_items) >= limit:
                        break
                        
                    tv_id = tv.get('id')
                    
                    # Dizi detaylarını al
                    details = get_tv_details(tv_id)
                    
                    if not details:
                        continue
                        
                    # Yeni öğe oluştur
                    new_item = {
                        'id': tv_id,
                        'title': details.get('name', ''),
                        'overview': details.get('overview', ''),
                        'release_date': details.get('first_air_date', ''),
                        'content_type': 'tv',
                        'poster_path': details.get('poster_path', ''),
                        'tmdb_details': json.dumps(details)
                    }
                    
                    new_items.append(new_item)
                    logger.info(f"Dizi yüklendi: {new_item['title']} (ID: {tv_id})")
                
                # Sonraki sayfa
                page += 1
                
                # API limiti aşılmaması için kısa bir duraklama
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Sayfa {page} yüklenirken hata: {str(e)}")
                break
                
        # Yeni DataFrame oluştur ve mevcut olanla birleştir
        if new_items:
            new_df = pd.DataFrame(new_items)
            
            if items_df is None or items_df.empty:
                items_df = new_df
            else:
                # Zaten var olan öğeleri çıkar
                existing_ids = set(items_df['id']) if 'id' in items_df.columns else set()
                new_df = new_df[~new_df['id'].isin(existing_ids)]
                
                # Birleştir
                items_df = pd.concat([items_df, new_df], ignore_index=True)
                
            # Dosyaya kaydet
            os.makedirs(os.path.dirname(ITEMS_DATA_PATH), exist_ok=True)
            items_df.to_csv(ITEMS_DATA_PATH, index=False)
            
            logger.info(f"{len(new_items)} dizi başarıyla yüklendi")
        else:
            logger.warning("Hiçbir dizi yüklenemedi")
            
    except Exception as e:
        logger.error(f"Popüler dizileri yükleme hatası: {str(e)}") 