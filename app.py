"""
MovieApp ML Backend - Ana Flask Uygulaması
Modern hibrit film öneri sistemi backend'i

Bu dosya Flask web uygulamasının ana giriş noktasıdır.
TMDb API entegrasyonu, makine öğrenmesi modelleri ve
kullanıcı etkileşimleri burada yönetilir.
"""

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

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.datetime.now().year}

@app.context_processor
def inject_user_watchlist():
    global user_watchlist
    return {'user_watchlist': user_watchlist}

CORS(app)  # CORS desteği ekle

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Yapılandırma
TMDB_API_KEY = os.getenv('TMDB_API_KEY', 'your_api_key_here')
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
user_watchlist = {}

# --- Flask Routes ---

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    """Arama sayfası"""
    query = request.args.get('q', '')
    if not query:
        return render_template('search.html')
    
    # TMDb API'dan arama yap
    movies = search_movies(query)
    tv_series = search_tv_series(query)
    
    return render_template('search.html', 
                         query=query, 
                         movies=movies, 
                         tv_series=tv_series)

@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    """Kullanıcı için öneriler API endpoint'i"""
    try:
        recommendations = get_ml_recommendations(user_id)
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"Öneri API hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/movies/<int:movie_id>/rate', methods=['POST'])
def rate_movie(movie_id):
    """Film puanlama endpoint'i"""
    try:
        data = request.get_json()
        rating = data.get('rating')
        user_id = data.get('user_id', 1)  # Varsayılan kullanıcı
        
        if not rating or rating < 1 or rating > 10:
            return jsonify({
                'success': False,
                'error': 'Geçersiz puan (1-10 arası olmalı)'
            }), 400
        
        # Puanı kaydet
        add_rating(user_id, movie_id, rating)
        
        return jsonify({
            'success': True,
            'message': 'Puan başarıyla kaydedildi'
        })
    except Exception as e:
        logger.error(f"Puanlama hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# --- Yardımcı Fonksiyonlar ---

def search_movies(query, page=1):
    """TMDb API'dan film arama"""
    try:
        url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'page': page,
            'language': 'tr-TR'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('results', [])
    except Exception as e:
        logger.error(f"Film arama hatası: {str(e)}")
        return []

def search_tv_series(query, page=1):
    """TMDb API'dan dizi arama"""
    try:
        url = f"https://api.themoviedb.org/3/search/tv"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'page': page,
            'language': 'tr-TR'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('results', [])
    except Exception as e:
        logger.error(f"Dizi arama hatası: {str(e)}")
        return []

def get_ml_recommendations(user_id, n=10):
    """Makine öğrenmesi tabanlı öneriler"""
    try:
        # Hibrit model varsa kullan
        if hybrid_model:
            return hybrid_model.recommend(user_id, n)
        
        # İçerik tabanlı model varsa kullan
        if cb_model and user_watchlist:
            watchlist_ids = list(user_watchlist.keys())
            return recommend_from_watchlist(cb_model, watchlist_ids, n)
        
        # Fallback: popüler içerikler
        return get_popular_recommendations(n)
    
    except Exception as e:
        logger.error(f"ML öneri hatası: {str(e)}")
        return []

def get_popular_recommendations(n=10):
    """Popüler içerik önerileri (fallback)"""
    try:
        url = f"https://api.themoviedb.org/3/movie/popular"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'tr-TR'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get('results', [])
        return results[:n]
    except Exception as e:
        logger.error(f"Popüler içerik hatası: {str(e)}")
        return []

def add_rating(user_id, item_id, rating, item_details=None):
    """Kullanıcı puanı ekleme"""
    global ratings_df
    
    try:
        new_rating = {
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if ratings_df is None or ratings_df.empty:
            ratings_df = pd.DataFrame([new_rating])
        else:
            ratings_df = pd.concat([ratings_df, pd.DataFrame([new_rating])], ignore_index=True)
        
        # Disk'e kaydet
        os.makedirs(os.path.dirname(RATINGS_DATA_PATH), exist_ok=True)
        ratings_df.to_csv(RATINGS_DATA_PATH, index=False)
        
        logger.info(f"Puan eklendi: User {user_id}, Item {item_id}, Rating {rating}")
        
    except Exception as e:
        logger.error(f"Puan ekleme hatası: {str(e)}")

# --- Uygulama Başlatma ---

if __name__ == '__main__':
    logger.info("MovieApp ML Backend başlatılıyor...")
    
    # Modelleri ve verileri yükle
    load_models()
    load_data()
    
    # Uygulamayı başlat
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )