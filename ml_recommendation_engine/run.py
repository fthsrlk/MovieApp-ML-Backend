"""
Öneri sistemini çalıştırmak için basit bir çalıştırıcı script
"""

import os
import sys
import logging

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Proje kök dizinini ayarla
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Film ve TV Dizisi Öneri Sistemi")
    parser.add_argument("--mode", type=str, choices=["api", "train", "recommend"], 
                        default="api", help="Çalıştırma modu")
    parser.add_argument("--user_id", type=int, default=1, help="Öneriler için kullanıcı ID")
    parser.add_argument("--limit", type=int, default=10, help="Öneri sayısı")
    parser.add_argument("--strategy", type=str, 
                        choices=["collaborative", "content_based", "hybrid"], 
                        default="hybrid", help="Öneri stratejisi")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        logger.info("API modu başlatılıyor...")
        from api.app import app
        
        # Yapılandırma değerlerini al
        host = os.getenv('API_HOST', '0.0.0.0')
        port = int(os.getenv('API_PORT', 5000))
        debug = os.getenv('API_DEBUG', 'True').lower() in ('true', '1', 't')
        
        # Flask uygulamasını başlat
        logger.info(f"API başlatılıyor: http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)
        
    elif args.mode == "train":
        logger.info("Model eğitimi başlatılıyor...")
        from main import train_models, load_data
        
        # Veri yükleme
        items_df, ratings_df = load_data()
        
        # Model eğitimi
        train_models(items_df, ratings_df, model_type="all")
        
    elif args.mode == "recommend":
        logger.info(f"Kullanıcı {args.user_id} için öneriler getiriliyor...")
        from main import get_recommendations, load_data, load_models
        
        # Veri ve modelleri yükle
        items_df, _ = load_data()
        cf_model, cb_model, hybrid_model = load_models()
        
        # Önerileri al
        recommendations = get_recommendations(
            args.user_id, 
            items_df, 
            model_type=args.strategy,
            n=args.limit
        )
        
        # Önerileri göster
        print("\nÖnerilen İçerikler:")
        print("-" * 80)
        for i, (item_id, title, score) in enumerate(recommendations, 1):
            print(f"{i}. {title} (ID: {item_id}, Skor: {score:.2f})")
        print("-" * 80) 