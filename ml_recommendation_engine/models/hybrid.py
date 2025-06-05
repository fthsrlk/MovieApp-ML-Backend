"""
Hibrit öneri algoritması
"""

import numpy as np
import pandas as pd
import logging
import joblib
import os
from .collaborative import CollaborativeFiltering
from .content_based import ContentBasedFiltering

logger = logging.getLogger(__name__)

class HybridRecommender:
    """
    Hibrit öneri sistemi
    
    İşbirlikçi filtreleme ve içerik tabanlı filtreleme yaklaşımlarını birleştirir.
    """
    
    def __init__(self, cf_model=None, cb_model=None, cf_weight=0.8, cb_weight=0.2):
        """
        Hibrit öneri modeli
        
        Args:
            cf_model: İşbirlikçi filtreleme modeli
            cb_model: İçerik tabanlı filtreleme modeli
            cf_weight (float): İşbirlikçi filtreleme ağırlığı
            cb_weight (float): İçerik tabanlı filtreleme ağırlığı
        """
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        
        # Ağırlıkların toplamı 1 olmalı
        total = cf_weight + cb_weight
        if total != 1.0:
            self.cf_weight = cf_weight / total
            self.cb_weight = cb_weight / total
            
        logger.info(f"Hibrit model oluşturuldu: CF ağırlığı={self.cf_weight:.2f}, CB ağırlığı={self.cb_weight:.2f}")
        
        # Modellerin yüklü olup olmadığını kontrol et
        self.cf_available = cf_model is not None
        self.cb_available = cb_model is not None
        
        self.normalize_scores = True
    
    def fit(self, ratings_df, items_df):
        """
        Modeli eğitir
        
        Args:
            ratings_df (pd.DataFrame): Kullanıcı-öğe değerlendirmeleri
            items_df (pd.DataFrame): Öğe özellikleri
            
        Returns:
            self
        """
        logger.info("HybridRecommender.fit başladı")
        
        try:
            # İşbirlikçi filtreleme modeli eğitimi
            logger.info("İşbirlikçi filtreleme modeli eğitiliyor")
            # Güvenli bir k değeri belirleyelim
            num_factors = min(10, min(len(ratings_df['user_id'].unique()), len(ratings_df['item_id'].unique())) - 1)
            if num_factors < 1:
                num_factors = 1
                
            self.cf_model = CollaborativeFiltering(
                method='matrix-factorization',
                num_factors=num_factors
            )
            self.cf_model.fit(ratings_df)
            
            # İçerik tabanlı model eğitimi
            logger.info("İçerik tabanlı model eğitiliyor")
            self.cb_model = ContentBasedFiltering()
            self.cb_model.fit(items_df)
            
            # Değerlendirme dataframe'ini veri yapılarımızda sakla
            self.ratings_df = ratings_df.copy()
            self.items_df = items_df.copy()
            
            self.is_fitted = True
            logger.info("HybridRecommender.fit tamamlandı")
            
            return self
            
        except Exception as e:
            logger.error(f"Model eğitimi hatası: {str(e)}")
            raise
    
    def recommend(self, user_id, n=10, exclude_watched=True, ratings_df=None):
        """
        Kullanıcı için hibrit öneriler oluştur
        
        Args:
            user_id: Kullanıcı ID
            n (int): Önerilecek öğe sayısı
            exclude_watched (bool): İzlenmiş öğeleri hariç tut
            ratings_df (pd.DataFrame, optional): İzlenen içerikleri belirlemek için
                değerlendirme verileri
            
        Returns:
            list: (item_id, score) ikililerinden oluşan liste
        """
        logger.info(f"Kullanıcı {user_id} için hibrit öneriler oluşturuluyor")
        
        # Kullanıcı değerlendirme sayısına göre dinamik ağırlık
        if ratings_df is not None:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            user_rating_count = len(user_ratings)
            
            # Değerlendirme sayısına göre ağırlıkları dinamik olarak ayarla
            if user_rating_count < 3:
                # Çok az değerlendirme - içerik tabanlı önerilere ağırlık ver
                dyn_cf_weight, dyn_cb_weight = 0.15, 0.85
            elif user_rating_count < 5:
                # Az değerlendirme - içerik tabanlı önerilere biraz daha fazla ağırlık
                dyn_cf_weight, dyn_cb_weight = 0.3, 0.7
            elif user_rating_count < 10:
                # Orta seviye - dengeli karma
                dyn_cf_weight, dyn_cb_weight = 0.5, 0.5
            elif user_rating_count < 20:
                # İyi seviye - işbirlikçi filtrelemeye biraz daha fazla ağırlık
                dyn_cf_weight, dyn_cb_weight = 0.7, 0.3
            else:
                # Çok fazla değerlendirme - işbirlikçi filtrelemeye ağırlık ver
                dyn_cf_weight, dyn_cb_weight = 0.85, 0.15
            
            # Kullanıcının değerlendirme desenlerini analiz et
            genre_bias = False
            if user_rating_count >= 3:
                try:
                    # Kullanıcının değerlendirdiği tür dağılımını analiz et
                    genre_counts = {}
                    total_ratings = 0
                    
                    for _, rating_row in user_ratings.iterrows():
                        item_id = rating_row['item_id']
                        
                        # items_df varsa tür bilgisini al
                        if hasattr(self, 'items_df'):
                            item_data = self.items_df[self.items_df['item_id'] == item_id]
                            if not item_data.empty and 'genres' in item_data.columns:
                                genres = item_data.iloc[0]['genres']
                                if isinstance(genres, str) and genres:
                                    for genre in genres.split(','):
                                        genre = genre.strip()
                                        if genre:
                                            genre_counts[genre] = genre_counts.get(genre, 0) + 1
                                            total_ratings += 1
                    
                    # Tür bazında bir bias var mı?
                    if total_ratings > 0:
                        for genre, count in genre_counts.items():
                            if count / total_ratings > 0.5:  # Tek bir türün >%50 olması bias gösterir
                                genre_bias = True
                                # Tür biası varsa, içerik tabanlı filtrelemeye biraz daha ağırlık ver
                                dyn_cb_weight = min(0.9, dyn_cb_weight + 0.1)
                                dyn_cf_weight = 1.0 - dyn_cb_weight
                                break
                except Exception as e:
                    logger.warning(f"Tür analizi sırasında hata: {str(e)}")
            
            logger.info(f"Kullanıcı {user_id} için dinamik ağırlıklar: CF={dyn_cf_weight:.2f}, CB={dyn_cb_weight:.2f}" +
                       (", Tür Biası Tespit Edildi" if genre_bias else ""))
        else:
            dyn_cf_weight, dyn_cb_weight = self.cf_weight, self.cb_weight
        
        # İzlenen içerikleri belirle
        watched = set()
        if exclude_watched and ratings_df is not None:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            watched = set(user_ratings['item_id'].tolist())
        
        # Kullanıcı içerik tabanlı modelde bulunmuyorsa ve yeterli veri varsa
        # dinamik olarak profil oluştur
        if ratings_df is not None and user_id not in self.cb_model.user_profiles:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            if len(user_ratings) > 0:
                logger.info(f"Hibrit öneriler için kullanıcı {user_id} profili oluşturuluyor")
                # Kullanıcı için profil oluştur
                if hasattr(self.cb_model, '_create_user_profile_for_single_user'):
                    self.cb_model._create_user_profile_for_single_user(user_id, user_ratings)
                    logger.info(f"Kullanıcı {user_id} için profil güncellendi")
        
        # Her iki modelden öneriler al - çok daha fazla öneri iste
        expanded_n = max(50, n * 5)  # En az 50 öneri iste veya n'in 5 katı
        
        cf_recommendations = self.cf_model.recommend(user_id, n=expanded_n, exclude_watched=exclude_watched)
        cb_recommendations = self.cb_model.recommend_for_user(
            user_id, n=expanded_n, exclude_watched=exclude_watched, ratings_df=ratings_df
        )
        
        # Önerileri birleştir
        item_scores = {}
        
        # İşbirlikçi filtreleme önerilerini ekle
        for item_id, score in cf_recommendations:
            if item_id in watched:
                continue
            item_scores[item_id] = dyn_cf_weight * score
        
        # İçerik tabanlı önerileri ekle
        for item_id, score in cb_recommendations:
            if item_id in watched:
                continue
            if item_id in item_scores:
                item_scores[item_id] += dyn_cb_weight * score
            else:
                item_scores[item_id] = dyn_cb_weight * score
        
        # Önerilen içerik tür çeşitliliğini sağla
        genre_diversity = {}
        if hasattr(self, 'items_df'):
            try:
                # Her bir önerilen öğenin türünü belirle
                for item_id in item_scores.keys():
                    item_data = self.items_df[self.items_df['item_id'] == item_id]
                    if not item_data.empty and 'genres' in item_data.columns:
                        genres = item_data.iloc[0]['genres']
                        if isinstance(genres, str) and genres:
                            genre_list = [g.strip() for g in genres.split(',') if g.strip()]
                            if genre_list:
                                # Tür bilgisini sakla
                                genre_diversity[item_id] = genre_list
            except Exception as e:
                logger.warning(f"Tür çeşitliliği analizi sırasında hata: {str(e)}")
        
        # Puanları normalize et
        if self.normalize_scores and item_scores:
            max_score = max(item_scores.values())
            if max_score > 0:
                for item_id in item_scores:
                    item_scores[item_id] /= max_score
        
        # Sonuçları sırala
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Çeşitlilik için geliştirilmiş strateji: 
        # 1. En yüksek puanlı %30 öneriyi al
        # 2. Orta aralıktan %40'ı tür çeşitliliğine göre seç
        # 3. Kalan %30'u daha geniş bir havuzdan rastgele seç
        top_percent = 0.3
        mid_percent = 0.4
        random_percent = 0.3
        
        top_count = max(1, int(n * top_percent))  # En az 1 tane olsun
        mid_count = max(1, int(n * mid_percent))  # En az 1 tane olsun
        random_count = n - top_count - mid_count
            
        # İlk grup - en yüksek puanlı öğeler
        top_items = sorted_items[:top_count]
        top_item_ids = {item[0] for item in top_items}
        
        # İkinci grup - tür çeşitliliği için
        mid_pool = [item for item in sorted_items[top_count:min(len(sorted_items), expanded_n // 2)] 
                   if item[0] not in top_item_ids]
        
        # Şu ana kadar seçilmiş türleri takip et
        selected_genres = set()
        for item_id, _ in top_items:
            if item_id in genre_diversity:
                selected_genres.update(genre_diversity[item_id])
        
        # Tür çeşitliliğine dayalı seçim
        mid_items = []
        remaining_mid_pool = mid_pool.copy()
        
        # Önce tür çeşitliliğini artıracak öğeleri seç
        for i in range(mid_count):
            if not remaining_mid_pool:
                break
                
            # En çok yeni tür getiren öğeyi bul
            best_item = None
            best_new_genres = -1
            
            for item in remaining_mid_pool:
                item_id = item[0]
                if item_id in genre_diversity:
                    item_genres = set(genre_diversity[item_id])
                    new_genres = len(item_genres - selected_genres)
                    if new_genres > best_new_genres:
                        best_new_genres = new_genres
                        best_item = item
            
            if best_item:
                mid_items.append(best_item)
                remaining_mid_pool.remove(best_item)
                
                # Seçilen türleri güncelle
                item_id = best_item[0]
                if item_id in genre_diversity:
                    selected_genres.update(genre_diversity[item_id])
            else:
                # Yeni tür getiren öğe yoksa rastgele seç
                import random
                idx = random.randint(0, len(remaining_mid_pool) - 1)
                mid_items.append(remaining_mid_pool[idx])
                remaining_mid_pool.pop(idx)
        
        # Eksik kalan mid_items'ı tamamla
        while len(mid_items) < mid_count and remaining_mid_pool:
            mid_items.append(remaining_mid_pool.pop(0))
        
        # Üçüncü grup - çeşitlilik için rastgele seçim
        # top_items ve mid_items dışındaki tüm öğelerden oluşan havuz
        selected_ids = {item[0] for item in top_items + mid_items}
        diversity_pool = [item for item in sorted_items if item[0] not in selected_ids]
        
        # Rastgele seçim için ağırlıklı olasılık hesapla
        random_items = []
        
        if diversity_pool and random_count > 0:
            import random
            import numpy as np
            
            # Ağırlıklı seçim için puanları dizi haline getir
            pool_items = [item[0] for item in diversity_pool]
            pool_scores = np.array([item[1] for item in diversity_pool])
            
            # Puanları olasılıklara dönüştür (daha yüksek puanlı öğelerin seçilme olasılığı daha yüksek)
            if np.sum(pool_scores) > 0:
                probs = pool_scores / np.sum(pool_scores)
                
                # Tekrarsız seçim yap (np.random.choice ile)
                try:
                    # Havuzda yeteri kadar eleman yoksa, mevcut tüm elemanları seç
                    size = min(random_count, len(pool_items))
                    selected_indices = np.random.choice(
                        len(pool_items), 
                        size=size, 
                        replace=False, 
                        p=probs
                    )
                    for idx in selected_indices:
                        random_items.append((pool_items[idx], pool_scores[idx]))
                except Exception as e:
                    # Hata durumunda basit rastgele seçim yap
                    logger.warning(f"Rastgele seçim sırasında hata: {str(e)}, basit rastgele seçim yapılıyor")
                    random_indices = random.sample(range(len(pool_items)), min(random_count, len(pool_items)))
                    for idx in random_indices:
                        random_items.append((pool_items[idx], pool_scores[idx]))
            else:
                # Eğer tüm puanlar sıfırsa, rastgele seç
                random_indices = random.sample(range(len(pool_items)), min(random_count, len(pool_items)))
                for idx in random_indices:
                    random_items.append((pool_items[idx], pool_scores[idx]))
        
        # Sonuçları birleştir
        result = top_items + mid_items + random_items
        
        # Sonuçlar yeterli değilse, eksik kalan kısımları sorted_items'dan tamamla
        if len(result) < n and len(sorted_items) > len(result):
            result_ids = {item[0] for item in result}
            extra_items = [item for item in sorted_items if item[0] not in result_ids]
            result.extend(extra_items[:n-len(result)])
        
        logger.info(f"Kullanıcı {user_id} için toplam {len(result)} öneri oluşturuldu")
        return result[:n]
    
    def get_similar_items(self, item_id, n=10):
        """
        Belirli bir öğeye benzer öğeleri döndürür
        
        Args:
            item_id: Öğe ID
            n (int): Sonuç sayısı
            
        Returns:
            list: (item_id, similarity_score) ikililerinden oluşan liste
        """
        # İçerik tabanlı benzerlik kullan
        return self.cb_model.get_similar_items(item_id, n)
    
    def explain_recommendation(self, user_id, item_id, ratings_df=None):
        """
        Bir öneri için açıklama oluşturur
        
        Args:
            user_id: Kullanıcı ID
            item_id: Öğe ID
            ratings_df (pd.DataFrame, optional): Kullanıcı değerlendirmeleri
            
        Returns:
            dict: Açıklama bilgileri
        """
        explanation = {
            'user_id': user_id,
            'item_id': item_id,
            'cf_score': 0,
            'cb_score': 0,
            'similar_items': [],
            'similar_users': []
        }
        
        # İşbirlikçi filtreleme puanı
        if self.cf_model:
            explanation['cf_score'] = self.cf_model.predict(user_id, item_id)
        
        # İçerik tabanlı benzerlik
        if self.cb_model:
            # Benzer öğeler
            similar_items = self.cb_model.get_similar_items(item_id, n=5)
            explanation['similar_items'] = similar_items
            
            # Kullanıcı profili varsa, içerik tabanlı puan
            if user_id in self.cb_model.user_profiles:
                # Kullanıcının izlediği içerikler
                if ratings_df is not None:
                    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
                    explanation['cb_score'] = self.cb_model.recommend_for_user(
                        user_id, n=100, ratings_df=ratings_df
                    )
                    
                    # Önerilen öğenin sıralamasını bul
                    for i, (rec_item_id, score) in enumerate(explanation['cb_score']):
                        if rec_item_id == item_id:
                            explanation['cb_rank'] = i + 1
                            explanation['cb_score'] = score
                            break
        
        return explanation
    
    def save(self, filepath):
        """
        Modeli kaydet
        
        Args:
            filepath (str): Kayıt yolu
        """
        joblib.dump(self, filepath)
        logger.info(f"Model kaydedildi: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Modeli yükle
        
        Args:
            filepath (str): Kayıt yolu
            
        Returns:
            HybridRecommender: Yüklenen model
        """
        if os.path.exists(filepath):
            logger.info(f"Model yükleniyor: {filepath}")
            return joblib.load(filepath)
        else:
            logger.warning(f"Model bulunamadı: {filepath}")
            return None 