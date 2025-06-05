"""
ML Öneri Sistemi Ana Modülü
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Modül yolunu ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Ortam değişkenlerini yükle
load_dotenv()

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir, "recommendation_engine.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Veri ve model yolları
MODEL_DIR = os.path.join(current_dir, "models")
DATA_DIR = os.path.join(current_dir, "data")
CACHE_DIR = os.path.join(current_dir, "cache")

# Dizinleri oluştur
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Model dosya yolları
CF_MODEL_PATH = os.path.join(MODEL_DIR, "collaborative_model.pkl")
CB_MODEL_PATH = os.path.join(MODEL_DIR, "content_based_model.pkl")
HYBRID_MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_model.pkl")

# Veri dosya yolları
ITEMS_PATH = os.path.join(DATA_DIR, "items.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")

def load_or_create_data(force_reload=False):
    """
    Veri setlerini yükle veya oluştur
    
    Args:
        force_reload (bool): Mevcut veriler olsa bile yeniden yükle
        
    Returns:
        tuple: (items_df, ratings_df)
    """
    # Mevcut veri setleri varsa yükle (force_reload=False ise)
    if not force_reload and os.path.exists(ITEMS_PATH) and os.path.exists(RATINGS_PATH):
        try:
            items_df = pd.read_csv(ITEMS_PATH)
            ratings_df = pd.read_csv(RATINGS_PATH)
            
            # Veri boyutlarını kontrol et
            if len(items_df) > 10 and len(ratings_df) > 10:
                logger.info(f"Mevcut veri setleri yüklendi: {len(items_df)} içerik, {len(ratings_df)} değerlendirme")
                return items_df, ratings_df
        except Exception as e:
            logger.error(f"Veri yükleme hatası: {str(e)}")
    
    # TMDB'den veri yükleme
    try:
        from data.loader import TMDBDataLoader
        
        api_key = os.getenv("TMDB_API_KEY", "3bc6092717beedc3a05e4d0809435ef2")
        loader = TMDBDataLoader(api_key, language="tr-TR")
        
        # Örnek veri oluştur - daha fazla ve daha kaliteli veri için parametreleri artırdık
        items_df, ratings_df = loader.load_sample_data(
            n_movies=5000, 
            n_tv=3000, 
            min_vote_count=200  # Biraz daha seçici olalım ama yine de yeterli veri alalım
        )
        
        # İçerik verilerini işle
        from data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        items_df = preprocessor.preprocess_items(items_df)
        
        # Verileri kaydet
        items_df.to_csv(ITEMS_PATH, index=False)
        ratings_df.to_csv(RATINGS_PATH, index=False)
        
        logger.info(f"Yeni veri setleri oluşturuldu: {len(items_df)} içerik, {len(ratings_df)} değerlendirme")
        return items_df, ratings_df
        
    except Exception as e:
        logger.error(f"Veri oluşturma hatası: {str(e)}")
        # Varsayılan boş DataFrame'ler oluştur
        items_df = pd.DataFrame(columns=['item_id', 'title', 'overview', 'release_date', 'content_type', 'poster_path'])
        ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
        return items_df, ratings_df

def train_and_save_models(items_df, ratings_df, force_retrain=False):
    """
    Modelleri eğit ve kaydet
    
    Args:
        items_df (pd.DataFrame): İçerik verileri
        ratings_df (pd.DataFrame): Değerlendirme verileri
        force_retrain (bool): Mevcut modeller olsa bile yeniden eğit
        
    Returns:
        tuple: (cf_model, cb_model, hybrid_model)
    """
    # Veri boyutlarını kontrol et
    if len(ratings_df) < 10:
        logger.warning("Yeterli değerlendirme verisi yok, model eğitimi atlanıyor")
        return None, None, None
    
    if len(items_df) < 10:
        logger.warning("Yeterli içerik verisi yok, model eğitimi atlanıyor")
        return None, None, None
    
    # Modelleri içe aktar
    from models.collaborative import CollaborativeFiltering
    from models.content_based import ContentBasedFiltering
    from models.hybrid import HybridRecommender
    
    # İşbirlikçi filtreleme modeli
    cf_model = None
    if force_retrain or not os.path.exists(CF_MODEL_PATH):
        try:
            logger.info("İşbirlikçi filtreleme modeli eğitiliyor")
            # En fazla 250 faktör kullan veya içerik/kullanıcı sayısına göre ayarla
            num_factors = min(250, min(len(ratings_df['user_id'].unique()), len(ratings_df['item_id'].unique())) - 1)
            if num_factors < 1:
                num_factors = 1
                
            cf_model = CollaborativeFiltering(
                method='matrix-factorization',
                num_factors=num_factors,
                reg_param=0.001  # Çok daha düşük düzenlileştirme parametresi
            )
            cf_model.fit(ratings_df)
            
            # Modeli kaydet
            cf_model.save(CF_MODEL_PATH)
            logger.info(f"İşbirlikçi filtreleme modeli kaydedildi: {CF_MODEL_PATH}")
        except Exception as e:
            logger.error(f"İşbirlikçi filtreleme modeli eğitim hatası: {str(e)}")
    else:
        try:
            logger.info(f"Mevcut işbirlikçi filtreleme modeli yükleniyor: {CF_MODEL_PATH}")
            cf_model = CollaborativeFiltering.load(CF_MODEL_PATH)
        except Exception as e:
            logger.error(f"İşbirlikçi filtreleme modeli yükleme hatası: {str(e)}")
    
    # İçerik tabanlı filtreleme modeli
    cb_model = None
    if force_retrain or not os.path.exists(CB_MODEL_PATH):
        try:
            logger.info("İçerik tabanlı filtreleme modeli eğitiliyor")
            cb_model = ContentBasedFiltering(
                use_tfidf=True,  # TF-IDF ağırlıklandırma kullan
                min_rating=2.0   # Daha düşük minimum değerlendirme eşiği
            )
            cb_model.fit(items_df)
            
            # Kullanıcı profilleri oluşturma kısmını _create_user_profiles içinde yapacağız
            if len(ratings_df) > 0:
                logger.info("Kullanıcı profilleri oluşturuluyor")
                for user_id in ratings_df['user_id'].unique():
                    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
                    if len(user_ratings) > 0:
                        cb_model._create_user_profile_for_single_user(user_id, user_ratings)
            
            # Modeli kaydet
            cb_model.save(CB_MODEL_PATH)
            logger.info(f"İçerik tabanlı filtreleme modeli kaydedildi: {CB_MODEL_PATH}")
        except Exception as e:
            logger.error(f"İçerik tabanlı filtreleme modeli eğitim hatası: {str(e)}")
    else:
        try:
            logger.info(f"Mevcut içerik tabanlı filtreleme modeli yükleniyor: {CB_MODEL_PATH}")
            cb_model = ContentBasedFiltering.load(CB_MODEL_PATH)
        except Exception as e:
            logger.error(f"İçerik tabanlı filtreleme modeli yükleme hatası: {str(e)}")
    
    # Hibrit model
    hybrid_model = None
    if (cf_model is not None and cb_model is not None) and (force_retrain or not os.path.exists(HYBRID_MODEL_PATH)):
        try:
            logger.info("Hibrit model oluşturuluyor")
            hybrid_model = HybridRecommender(cf_model=cf_model, cb_model=cb_model)
            
            # Modeli kaydet
            hybrid_model.save(HYBRID_MODEL_PATH)
            logger.info(f"Hibrit model kaydedildi: {HYBRID_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Hibrit model oluşturma hatası: {str(e)}")
    elif os.path.exists(HYBRID_MODEL_PATH):
        try:
            logger.info(f"Mevcut hibrit model yükleniyor: {HYBRID_MODEL_PATH}")
            hybrid_model = HybridRecommender.load(HYBRID_MODEL_PATH)
        except Exception as e:
            logger.error(f"Hibrit model yükleme hatası: {str(e)}")
    
    return cf_model, cb_model, hybrid_model

def demo_recommendations(user_id=1, n=10):
    """
    Demo önerileri göster
    
    Args:
        user_id (int): Öneri yapılacak kullanıcı ID
        n (int): Öneri sayısı
    """
    # Modelleri yükle
    from models.collaborative import CollaborativeFiltering
    from models.content_based import ContentBasedFiltering
    from models.hybrid import HybridRecommender
    
    cf_model = CollaborativeFiltering.load(CF_MODEL_PATH)
    cb_model = ContentBasedFiltering.load(CB_MODEL_PATH)
    hybrid_model = HybridRecommender.load(HYBRID_MODEL_PATH)
    
    # Veri setlerini yükle
    items_df = pd.read_csv(ITEMS_PATH)
    ratings_df = pd.read_csv(RATINGS_PATH)
    
    print(f"\nKullanıcı {user_id} için öneriler:")
    
    # İşbirlikçi filtreleme önerileri
    if cf_model:
        cf_recs = cf_model.recommend(user_id, n=n)
        print("\nİşbirlikçi Filtreleme Önerileri:")
        for i, (item_id, score) in enumerate(cf_recs, 1):
            item = items_df[items_df['item_id'] == item_id]
            if not item.empty:
                title = item.iloc[0]['title']
                content_type = item.iloc[0]['content_type']
                print(f"{i}. {title} ({content_type}) - Puan: {score:.2f}")
    
    # İçerik tabanlı öneriler
    if cb_model:
        cb_recs = cb_model.recommend_for_user(user_id, n=n, ratings_df=ratings_df)
        print("\nİçerik Tabanlı Öneriler:")
        for i, (item_id, score) in enumerate(cb_recs, 1):
            item = items_df[items_df['item_id'] == item_id]
            if not item.empty:
                title = item.iloc[0]['title']
                content_type = item.iloc[0]['content_type']
                print(f"{i}. {title} ({content_type}) - Puan: {score:.2f}")
    
    # Hibrit öneriler
    if hybrid_model:
        hybrid_recs = hybrid_model.recommend(user_id, n=n, ratings_df=ratings_df)
        print("\nHibrit Öneriler:")
        for i, (item_id, score) in enumerate(hybrid_recs, 1):
            item = items_df[items_df['item_id'] == item_id]
            if not item.empty:
                title = item.iloc[0]['title']
                content_type = item.iloc[0]['content_type']
                print(f"{i}. {title} ({content_type}) - Puan: {score:.2f}")

def main():
    """
    Ana işlev
    """
    logger.info("MovieApp ML Öneri Sistemi başlatıldı")
    
    # Komut satırı argümanlarını işle
    import argparse
    parser = argparse.ArgumentParser(description="MovieApp ML Öneri Sistemi")
    parser.add_argument("--reload-data", action="store_true", help="Veriyi yeniden yükle")
    parser.add_argument("--retrain-models", action="store_true", help="Modelleri yeniden eğit")
    parser.add_argument("--demo", action="store_true", help="Demo önerileri göster")
    args = parser.parse_args()
    
    # Veri yükleme
    items_df, ratings_df = load_or_create_data(force_reload=args.reload_data)
    
    # Model eğitimi
    cf_model, cb_model, hybrid_model = train_and_save_models(
        items_df, ratings_df, force_retrain=args.retrain_models
    )
    
    # Demo önerileri
    if args.demo:
        demo_recommendations()
    
    logger.info("MovieApp ML Öneri Sistemi tamamlandı")
    
if __name__ == "__main__":
    main() 