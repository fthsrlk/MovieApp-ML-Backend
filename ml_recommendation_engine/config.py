"""
Yapılandırma ayarları
"""

import os
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()

# Temel yapılandırma
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv('MODEL_DIR', os.path.join(BASE_DIR, 'models'))
DATA_DIR = os.getenv('DATA_DIR', os.path.join(BASE_DIR, 'data'))
CACHE_DIR = os.getenv('CACHE_DIR', os.path.join(BASE_DIR, 'cache'))
LOG_DIR = os.getenv('LOG_DIR', os.path.join(BASE_DIR, 'logs'))

# API yapılandırması
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 5000))
API_DEBUG = os.getenv('API_DEBUG', 'True').lower() in ('true', '1', 't')
SECRET_KEY = os.getenv('SECRET_KEY', 'gizli-anahtar')

# TMDB API yapılandırması
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
TMDB_LANGUAGE = os.getenv('TMDB_LANGUAGE', 'tr-TR')

# Model yapılandırması
CF_METHOD = os.getenv('CF_METHOD', 'matrix-factorization')
CF_NUM_FACTORS = int(os.getenv('CF_NUM_FACTORS', 100))
CF_REG_PARAM = float(os.getenv('CF_REG_PARAM', 0.1))

CB_USE_TFIDF = os.getenv('CB_USE_TFIDF', 'True').lower() in ('true', '1', 't')
CB_MIN_RATING = float(os.getenv('CB_MIN_RATING', 3.5))

HYBRID_CF_WEIGHT = float(os.getenv('HYBRID_CF_WEIGHT', 0.7))
HYBRID_CB_WEIGHT = float(os.getenv('HYBRID_CB_WEIGHT', 0.3))

# Veri yapılandırması
MIN_RATINGS_PER_USER = int(os.getenv('MIN_RATINGS_PER_USER', 5))
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))

# Dosya yolları
CF_MODEL_PATH = os.path.join(MODEL_DIR, 'collaborative_model.pkl')
CB_MODEL_PATH = os.path.join(MODEL_DIR, 'content_based_model.pkl')
HYBRID_MODEL_PATH = os.path.join(MODEL_DIR, 'hybrid_model.pkl')
ITEMS_DATA_PATH = os.path.join(DATA_DIR, 'items.csv')
RATINGS_DATA_PATH = os.path.join(DATA_DIR, 'ratings.csv')

# Dizinleri oluştur
for directory in [MODEL_DIR, DATA_DIR, CACHE_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True) 