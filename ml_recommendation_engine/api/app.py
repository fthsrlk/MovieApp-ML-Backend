"""
Öneri sistemi API uygulaması
"""

from flask import Flask, request, jsonify
import os
import logging
import json
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
import pandas as pd
import sys
import pickle
import numpy as np
import re

# Proje kök dizini ayarlaması
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Modülleri doğrudan import et
try:
    from models.collaborative import CollaborativeFiltering
    from models.content_based import ContentBasedFiltering
    from models.hybrid import HybridRecommender
    from data.loader import TMDBDataLoader
    from data.preprocessor import DataPreprocessor
except ImportError as e:
    print(f"Import hatası: {e}")
    logging.error(f"Import hatası: {e}")
    # Hata çıkışı yapma, hata mesajı göster ve devam et
    print("Modüller yüklenemedi. Uygulama sınırlı işlevsellikle devam edecek.")

# Ortam değişkenlerini yükle
load_dotenv()

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask uygulaması
app = Flask(__name__)

# Yapılandırma
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'gizli-anahtar')
app.config['TMDB_API_KEY'] = os.getenv('TMDB_API_KEY', '')
app.config['MODEL_DIR'] = os.getenv('MODEL_DIR', 'models')
app.config['DATA_DIR'] = os.getenv('DATA_DIR', 'data')

# Model ve veri yolları
CF_MODEL_PATH = os.path.join(app.config['MODEL_DIR'], 'collaborative_model.pkl')
CB_MODEL_PATH = os.path.join(app.config['MODEL_DIR'], 'content_based_model.pkl')
HYBRID_MODEL_PATH = os.path.join(app.config['MODEL_DIR'], 'hybrid_model.pkl')
ITEMS_DATA_PATH = os.path.join(app.config['DATA_DIR'], 'items.csv')
RATINGS_DATA_PATH = os.path.join(app.config['DATA_DIR'], 'ratings.csv')

# Dizinleri oluştur
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
os.makedirs(app.config['DATA_DIR'], exist_ok=True)

# Global değişkenler
cf_model = None
cb_model = None
hybrid_model = None
items_df = None
ratings_df = None

# NumPy değerlerini JSON serileştirilebilir tiplere dönüştürme yardımcı fonksiyonu
def convert_numpy_types(obj):
    """
    NumPy değerlerini JSON serileştirilebilir tiplere dönüştürür
    
    Args:
        obj: Dönüştürülecek nesne
        
    Returns:
        JSON serileştirilebilir tipe dönüştürülmüş nesne
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Metin temizleme fonksiyonu
def clean_text(text):
    """
    Metni JSON serileştirme için güvenli hale getirir
    
    Args:
        text (str): Temizlenecek metin
        
    Returns:
        str: Temizlenmiş metin
    """
    if not isinstance(text, str):
        return ""
    
    # Özel karakterleri temizle
    text = re.sub(r'[\n\r\t\\"]', ' ', text)
    # Çift boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    # Başındaki ve sonundaki boşlukları kaldır
    text = text.strip()
    
    return text

def load_models():
    """
    Modelleri yükle
    """
    global cf_model, cb_model, hybrid_model
    
    # İşbirlikçi filtreleme modeli
    if os.path.exists(CF_MODEL_PATH):
        cf_model = CollaborativeFiltering.load(CF_MODEL_PATH)
        logger.info("İşbirlikçi filtreleme modeli yüklendi")
    
    # İçerik tabanlı filtreleme modeli
    if os.path.exists(CB_MODEL_PATH):
        cb_model = ContentBasedFiltering.load(CB_MODEL_PATH)
        logger.info("İçerik tabanlı filtreleme modeli yüklendi")
    
    # Hibrit model
    if os.path.exists(HYBRID_MODEL_PATH):
        hybrid_model = HybridRecommender.load(HYBRID_MODEL_PATH)
        logger.info("Hibrit model yüklendi")

def load_data():
    """
    Verileri yükle
    """
    global items_df, ratings_df
    
    # İçerik verileri
    if os.path.exists(ITEMS_DATA_PATH):
        items_df = pd.read_csv(ITEMS_DATA_PATH)
        logger.info(f"İçerik verileri yüklendi: {len(items_df)} öğe")
    
    # Değerlendirme verileri
    if os.path.exists(RATINGS_DATA_PATH):
        ratings_df = pd.read_csv(RATINGS_DATA_PATH)
        logger.info(f"Değerlendirme verileri yüklendi: {len(ratings_df)} değerlendirme")

def generate_token(user_id):
    """
    JWT token oluştur
    
    Args:
        user_id: Kullanıcı ID
        
    Returns:
        str: JWT token
    """
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(days=1)
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    return token

def verify_token(token):
    """
    JWT token doğrula
    
    Args:
        token (str): JWT token
        
    Returns:
        dict: Token payload veya None
    """
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def auth_required(f):
    """
    Kimlik doğrulama gerektiren endpoint'ler için dekoratör
    """
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'Yetkilendirme başlığı eksik'}), 401
        
        try:
            token = auth_header.split(' ')[1]
        except IndexError:
            return jsonify({'error': 'Geçersiz yetkilendirme başlığı formatı'}), 401
        
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Geçersiz veya süresi dolmuş token'}), 401
        
        return f(*args, **kwargs)
    
    decorated.__name__ = f.__name__
    return decorated

@app.route('/api/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    """
    Kullanıcı için öneriler al
    """
    # Parametreleri al
    limit = request.args.get('limit', default=10, type=int)
    content_type = request.args.get('content_type', default=None)
    strategy = request.args.get('strategy', default='hybrid')
    
    # Verilerin yüklendiğinden emin ol
    if items_df is None or ratings_df is None:
        try:
            load_data()
            logger.info("Veriler yeniden yüklendi")
        except Exception as e:
            logger.error(f"Veri yükleme hatası: {str(e)}")
            return jsonify({'error': 'Veri yüklenirken hata oluştu'}), 500
    
    # Modellerin yüklendiğinden emin ol
    if (strategy == 'collaborative' and cf_model is None) or \
       (strategy == 'content_based' and cb_model is None) or \
       (strategy == 'hybrid' and hybrid_model is None):
        try:
            load_models()
            logger.info("Modeller yeniden yüklendi")
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return jsonify({'error': 'Modeller yüklenirken hata oluştu'}), 500
    
    # Modelleri kontrol et
    if strategy == 'collaborative' and cf_model is None:
        return jsonify({'error': 'İşbirlikçi filtreleme modeli yüklenmedi'}), 500
    
    if strategy == 'content_based' and cb_model is None:
        return jsonify({'error': 'İçerik tabanlı filtreleme modeli yüklenmedi'}), 500
    
    if strategy == 'hybrid' and hybrid_model is None:
        return jsonify({'error': 'Hibrit model yüklenmedi'}), 500
    
    # Önerileri al
    try:
        if strategy == 'collaborative':
            recommendations = cf_model.recommend(user_id, n=limit)
        elif strategy == 'content_based':
            # Güncel değerlendirme verileriyle kullanıcı profilini güncelleyerek öneri oluştur
            recommendations = cb_model.recommend_for_user(user_id, n=limit, ratings_df=ratings_df)
        else:  # hybrid
            # Hibrit modelde de güncel değerlendirme verilerini kullan
            recommendations = hybrid_model.recommend(user_id, n=limit, ratings_df=ratings_df)
        
        # İçerik tipine göre filtrele
        if content_type and items_df is not None:
            filtered_recommendations = []
            for item_id, score in recommendations:
                # NumPy değerlerini dönüştür
                item_id = convert_numpy_types(item_id)
                
                # İçerik bilgilerini bul ve filtreleme yap
                item_rows = items_df[items_df['item_id'] == item_id]
                
                if not item_rows.empty:
                    item_info = item_rows.iloc[0]
                    # content_type sütunu varsa ve belirtilen değere sahipse ekle
                    if 'content_type' in item_info and item_info['content_type'] == content_type:
                        filtered_recommendations.append((item_id, score))
                    # type sütunu varsa ve belirtilen değere sahipse ekle
                    elif 'type' in item_info and item_info['type'] == content_type:
                        filtered_recommendations.append((item_id, score))
                    # media_type sütunu varsa ve belirtilen değere sahipse ekle
                    elif 'media_type' in item_info and item_info['media_type'] == content_type:
                        filtered_recommendations.append((item_id, score))
            
            # Filtrelenmiş önerileri kullan
            recommendations = filtered_recommendations[:limit]
        
        # Önerileri zenginleştir
        results = []
        for item_id, score in recommendations:
            # NumPy değerlerini dönüştür
            item_id = convert_numpy_types(item_id)
            score = convert_numpy_types(score)
            
            # İçerik bilgilerini bul
            item_info = {}
            if items_df is not None:
                item_rows = items_df[items_df['item_id'] == item_id]
                if not item_rows.empty:
                    item_info = item_rows.iloc[0].to_dict()
            
            # Güvenli bir şekilde dönüştür
            cleaned_info = {}
            for key, value in item_info.items():
                # NaN değerlerini ele al
                if key in ['overview', 'title', 'poster_path'] and (pd.isna(value) or value is None):
                    cleaned_info[key] = ""
                elif pd.isna(value) or value is None:
                    continue  # NaN değerlerini atla
                else:
                    cleaned_info[key] = convert_numpy_types(value)
                    
                    # Metin alanlarını temizle
                    if isinstance(value, str):
                        cleaned_info[key] = clean_text(value)
            
            # Yanıtı istemci uyumlu formatta oluştur
            result = {
                'item_id': item_id,
                'score': score,
                'title': cleaned_info.get('title', ""),
                'content_type': cleaned_info.get('content_type', ""),
                'poster_path': cleaned_info.get('poster_path', ""),
                'overview': cleaned_info.get('overview', "")
            }
            
            results.append(result)
        
        return jsonify({
            'user_id': user_id,
            'strategy': strategy,
            'recommendations': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Öneri oluşturma hatası: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Öneriler oluşturulurken hata oluştu'}), 500

@app.route('/api/similar/<int:item_id>', methods=['GET'])
def get_similar_items(item_id):
    """
    Benzer öğeleri al
    """
    # Parametreleri al
    limit = request.args.get('limit', default=10, type=int)
    
    # Modeli kontrol et
    if cb_model is None:
        return jsonify({'error': 'İçerik tabanlı filtreleme modeli yüklenmedi'}), 500
    
    # Benzer öğeleri al
    try:
        similar_items = cb_model.get_similar_items(item_id, n=limit)
        
        # Sonuçları zenginleştir
        results = []
        for similar_id, score in similar_items:
            # NumPy değerlerini dönüştür
            similar_id = convert_numpy_types(similar_id)
            score = convert_numpy_types(score)
            
            # İçerik bilgilerini bul
            item_info = items_df[items_df['item_id'] == similar_id].iloc[0].to_dict() if similar_id in items_df['item_id'].values else {}
            
            # Dict içindeki NumPy değerlerini dönüştür
            item_info = {k: convert_numpy_types(v) for k, v in item_info.items()}
            
            # Metinleri temizle
            title = clean_text(item_info.get('title', ''))
            overview = clean_text(item_info.get('overview', ''))
            poster_path = clean_text(item_info.get('poster_path', ''))
            
            # Sonuç
            result = {
                'item_id': similar_id,
                'score': score,
                'title': title,
                'content_type': item_info.get('content_type', ''),
                'poster_path': poster_path,
                'overview': overview
            }
            results.append(result)
        
        return jsonify({
            'item_id': item_id,
            'similar_items': results
        })
    
    except Exception as e:
        logger.error(f"Benzer öğe hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

def train_models():
    """
    Modelleri eğit
    """
    global cf_model, cb_model, hybrid_model
    
    # Verileri kontrol et
    if items_df is None or ratings_df is None:
        logger.error("Veriler yüklenmedi, modeller eğitilemiyor")
        return False
    
    try:
        # İşbirlikçi filtreleme modeli
        cf_model = CollaborativeFiltering(
            method='als',
            num_factors=100,
            reg_param=0.1
        )
        cf_model.fit(ratings_df)
        
        # İçerik tabanlı filtreleme modeli
        cb_model = ContentBasedFiltering(min_rating=3.0)
        cb_model.fit(items_df, ratings_df)
        
        # Hibrit model
        hybrid_model = HybridRecommender(
            cf_model=cf_model,
            cb_model=cb_model,
            cf_weight=0.3,  # İşbirlikçi filtreleme ağırlığı azaltıldı
            cb_weight=0.7   # İçerik tabanlı filtreleme ağırlığı artırıldı
        )
        
        # Modelleri kaydet
        save_model(cf_model, CF_MODEL_PATH)
        save_model(cb_model, CB_MODEL_PATH)
        save_model(hybrid_model, HYBRID_MODEL_PATH)
        
        logger.info("Modeller başarıyla eğitildi ve kaydedildi")
        return True
    
    except Exception as e:
        logger.error(f"Model eğitimi hatası: {str(e)}")
        return False

@app.route('/api/train', methods=['POST'])
@auth_required
def train_models_endpoint():
    """
    Modelleri eğit API endpoint'i
    """
    # Parametreleri al
    data = request.json
    model_type = data.get('model_type', 'all')
    params = data.get('params', {})
    
    # Verileri kontrol et
    if items_df is None or ratings_df is None:
        return jsonify({'error': 'Veriler yüklenmedi'}), 500
    
    # Modelleri eğit
    try:
        result = train_models()
        if result:
            return jsonify({'success': True, 'message': 'Modeller başarıyla eğitildi'})
        else:
            return jsonify({'error': 'Model eğitimi başarısız oldu'}), 500
    
    except Exception as e:
        logger.error(f"Model eğitim hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_sample_data', methods=['POST'])
@auth_required
def load_sample_data():
    """
    Örnek veri yükle
    """
    # Parametreleri al
    data = request.json
    n_movies = data.get('n_movies', 100)
    n_tv = data.get('n_tv', 100)
    
    # TMDB API anahtarını kontrol et
    if not app.config['TMDB_API_KEY']:
        return jsonify({'error': 'TMDB API anahtarı bulunamadı'}), 500
    
    # Verileri yükle
    try:
        global items_df, ratings_df
        
        # TMDB veri yükleyici
        loader = TMDBDataLoader(api_key=app.config['TMDB_API_KEY'])
        
        # Örnek veri yükle
        items_df, ratings_df = loader.load_sample_data(n_movies=n_movies, n_tv=n_tv)
        
        # Verileri kaydet
        items_df.to_csv(ITEMS_DATA_PATH, index=False)
        ratings_df.to_csv(RATINGS_DATA_PATH, index=False)
        
        logger.info(f"Örnek veriler yüklendi: {len(items_df)} içerik, {len(ratings_df)} değerlendirme")
        
        return jsonify({
            'success': True,
            'message': 'Örnek veriler başarıyla yüklendi',
            'items_count': len(items_df),
            'ratings_count': len(ratings_df)
        })
    
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth', methods=['POST'])
def authenticate():
    """
    Kimlik doğrulama
    """
    # Parametreleri al
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    # Basit kimlik doğrulama (gerçek uygulamada daha güvenli olmalı)
    if username == 'admin' and password == 'password':
        token = generate_token(user_id=1)
        return jsonify({'token': token, 'user_id': 1})
    
    return jsonify({'error': 'Geçersiz kullanıcı adı veya şifre'}), 401

@app.route('/api/ratings', methods=['POST'])
def add_rating():
    """
    Kullanıcı değerlendirmesini ekle
    """
    global ratings_df
    
    # Parametreleri al
    data = request.json
    user_id = data.get('user_id')
    item_id = data.get('item_id')
    rating = data.get('rating')
    
    # Parametreleri kontrol et
    if not all([user_id, item_id, rating]):
        return jsonify({'error': 'Eksik parametreler: user_id, item_id ve rating gereklidir'}), 400
    
    try:
        # Değerlendirme ekle/güncelle
        # Aynı kullanıcı-öğe değerlendirmesi varsa güncelle
        if ratings_df is None:
            ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # Aynı kullanıcı-öğe değerlendirmesini filtrele
        mask = (ratings_df['user_id'] == user_id) & (ratings_df['item_id'] == item_id)
        if mask.any():
            # Varsa güncelle
            ratings_df.loc[mask, 'rating'] = rating
            ratings_df.loc[mask, 'timestamp'] = pd.Timestamp.now().timestamp()
        else:
            # Yoksa ekle
            new_rating = pd.DataFrame({
                'user_id': [user_id],
                'item_id': [item_id],
                'rating': [rating],
                'timestamp': [pd.Timestamp.now().timestamp()]
            })
            ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
        
        # Verileri kaydet
        ratings_df.to_csv(RATINGS_DATA_PATH, index=False)
        
        # Başarılı yanıt
        return jsonify({
            'success': True,
            'message': 'Değerlendirme başarıyla kaydedildi',
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating
        })
    
    except Exception as e:
        logger.error(f"Değerlendirme hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_user_model', methods=['POST'])
def train_user_model():
    """
    Belirli bir kullanıcı için modeli güncelle ve eğit
    """
    global hybrid_model
    
    # Parametreleri al
    data = request.json
    user_id = data.get('user_id')
    
    # Parametreleri kontrol et
    if not user_id:
        return jsonify({'error': 'Eksik parametre: user_id gereklidir'}), 400
    
    # Verileri kontrol et
    if items_df is None or ratings_df is None:
        return jsonify({'error': 'Veriler yüklenmedi'}), 500
    
    # Kullanıcının değerlendirmelerini kontrol et
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    if len(user_ratings) == 0:
        return jsonify({'error': 'Kullanıcının değerlendirmesi bulunmamaktadır'}), 400
    
    # Modeli eğit
    try:
        # Hibrit model
        hybrid_model = HybridRecommender()
        hybrid_model.fit(ratings_df, items_df)
        hybrid_model.save(HYBRID_MODEL_PATH)
        
        # Başarılı yanıt
        return jsonify({
            'success': True,
            'message': 'Model başarıyla eğitildi',
            'user_id': user_id,
            'ratings_count': len(user_ratings)
        })
    
    except Exception as e:
        logger.error(f"Model eğitim hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Sağlık kontrolü
    """
    return jsonify({
        'status': 'ok',
        'models': {
            'collaborative': cf_model is not None,
            'content_based': cb_model is not None,
            'hybrid': hybrid_model is not None
        },
        'data': {
            'items': items_df is not None,
            'ratings': ratings_df is not None
        }
    })

@app.route('/api/user_content', methods=['POST'])
def add_user_content():
    """
    Kullanıcının eklediği içeriği öneri sistemine dahil et
    """
    global items_df, ratings_df
    
    # Parametreleri al
    data = request.json
    user_id = data.get('user_id')
    content = data.get('content', {})
    
    # Parametreleri kontrol et
    if not user_id or not content:
        return jsonify({'error': 'Eksik parametreler: user_id ve content gereklidir'}), 400
    
    required_fields = ['item_id', 'title', 'content_type']
    for field in required_fields:
        if field not in content:
            return jsonify({'error': f'İçerik için eksik alan: {field}'}), 400
    
    try:
        # İçeriği items_df'e ekle/güncelle
        if items_df is None:
            items_df = pd.DataFrame(columns=['item_id', 'title', 'content_type', 'overview', 'poster_path', 'genres'])
        
        item_id = content['item_id']
        
        # Aynı item_id varsa güncelle
        if item_id in items_df['item_id'].values:
            for key, value in content.items():
                if key in items_df.columns:
                    items_df.loc[items_df['item_id'] == item_id, key] = value
        else:
            # Yeni içerik oluştur
            new_item = {col: content.get(col, '') for col in items_df.columns}
            new_item['item_id'] = item_id
            items_df = pd.concat([items_df, pd.DataFrame([new_item])], ignore_index=True)
        
        # İçeriği kaydet
        items_df.to_csv(ITEMS_DATA_PATH, index=False)
        
        # Kullanıcının içeriği beğendiğini varsayalım (ratings_df)
        # Değerlendirme ekle (eğer yoksa)
        if ratings_df is None:
            ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # Bu içerik için kullanıcının değerlendirmesi var mı kontrol et
        mask = (ratings_df['user_id'] == user_id) & (ratings_df['item_id'] == item_id)
        if not mask.any():
            # Yoksa ekle
            new_rating = pd.DataFrame({
                'user_id': [user_id],
                'item_id': [item_id],
                'rating': [5.0],  # Kullanıcı eklediği için en yüksek puan
                'timestamp': [pd.Timestamp.now().timestamp()]
            })
            ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
            ratings_df.to_csv(RATINGS_DATA_PATH, index=False)
        
        # Başarılı yanıt
        return jsonify({
            'success': True,
            'message': 'Kullanıcı içeriği başarıyla eklendi ve değerlendirildi',
            'user_id': user_id,
            'item_id': item_id
        })
    
    except Exception as e:
        logger.error(f"Kullanıcı içeriği ekleme hatası: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Uygulama başlatma
@app.before_first_request
def initialize():
    """
    İlk istek öncesi başlatma
    """
    load_models()
    load_data()

if __name__ == '__main__':
    # Modelleri ve verileri yükle
    load_data()
    load_models()
    
    # Uygulama host ve port bilgilerini ayarla
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 5000))
    debug = os.getenv('API_DEBUG', 'True').lower() in ('true', '1', 't')
    
    # Bilgi mesajı
    logger.info(f"API başlatılıyor: http://{host}:{port}")
    logger.info(f"Debug modu: {debug}")
    
    # Flask uygulamasını başlat
    app.run(host=host, port=port, debug=debug) 