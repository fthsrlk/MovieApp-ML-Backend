"""
İçerik tabanlı filtreleme (Content-based Filtering) algoritması
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import logging
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class ContentBasedFiltering:
    """
    İçerik tabanlı filtreleme modeli
    
    Öğelerin özelliklerine (meta verilere) dayalı benzerlik hesaplamaları
    yaparak öneri sunar.
    """
    
    def __init__(self, use_tfidf=True, min_rating=3.5):
        """
        Args:
            use_tfidf (bool): TF-IDF vektörizasyonu kullan
            min_rating (float): Bir içeriği beğenilmiş saymak için gereken minimum puan
        """
        self.use_tfidf = use_tfidf
        self.min_rating = min_rating
        self.vectorizer = None
        self.item_features = None
        self.item_ids = None
        self.item_map = {}  # item_id -> index
        self.rev_item_map = {}  # index -> item_id
        self.user_profiles = defaultdict(np.ndarray)
    
    def fit(self, items_df):
        """
        İçerik tabanlı filtreleme için TF-IDF modelini oluşturur.
        
        Args:
            items_df (pandas.DataFrame): Öğelerin bilgilerini içeren veri çerçevesi
                'item_id', 'content_type', 'title', 'overview' sütunları olması beklenir
        """
        self.items_df = items_df.copy()
        
        # Veri çerçevesini kontrol et
        if self.items_df.empty:
            print("Uyarı: Boş veri çerçevesi ile fit çağrısı yapıldı")
            return self
            
        # ID sütun adını kontrol et ve uyumlu hale getir
        if 'id' in self.items_df.columns and 'item_id' not in self.items_df.columns:
            self.items_df['item_id'] = self.items_df['id']
        elif 'item_id' in self.items_df.columns and 'id' not in self.items_df.columns:
            self.items_df['id'] = self.items_df['item_id']
            
        # Her iki ID sütunu da yoksa hata ver
        if 'id' not in self.items_df.columns and 'item_id' not in self.items_df.columns:
            raise ValueError("Veri çerçevesinde 'id' veya 'item_id' sütunu bulunamadı")
        
        # ID'lerin sayısal olmasını sağla
        id_column = 'id' if 'id' in self.items_df.columns else 'item_id'
        self.items_df[id_column] = self.items_df[id_column].astype('Int64')
        
        # Öğe kimliklerinin dizinini oluştur (item_id -> index eşleşmesi) 
        self.item_indices = {}
        for i, item_id in enumerate(self.items_df[id_column]):
            if pd.notna(item_id):
                self.item_indices[item_id] = i
        
        # Özellik metni oluştur - zengin ve detaylı
        self.items_df['feature_text'] = self.items_df.apply(self._create_feature_text, axis=1)
        
        # TF-IDF vektörlerini oluştur
        self.vectorizer = TfidfVectorizer(analyzer='word', 
                                         stop_words='english',
                                         ngram_range=(1, 2),  # 1-2 kelimeleri yakala (önemli)
                                         max_features=10000)  # Vektör boyutunu sınırla
        
        try:
            # Özellik matrisini oluştur
            self.tfidf_matrix = self.vectorizer.fit_transform(self.items_df['feature_text'])
            
            # Kosinüs benzerlik matrisini oluştur
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            
            print(f"İçerik tabanlı model oluşturuldu: {len(self.items_df)} öğe, {self.tfidf_matrix.shape[1]} özellik")
        except Exception as e:
            print(f"TF-IDF matrisi oluşturulurken hata: {str(e)}")
            # Boş matrisi oluştur
            self.tfidf_matrix = None
            self.cosine_sim = np.zeros((len(self.items_df), len(self.items_df)))
        
        return self
    
    def _create_feature_text(self, row):
        """
        Öğelerin özelliklerini tek bir metin alanında birleştirir ve zenginleştirir
        
        Args:
            row: DataFrame satırı
            
        Returns:
            str: Birleştirilmiş özellik metni
        """
        # Kullanılabilecek sütunlar 
        text_columns = ['title', 'overview', 'genres', 'keywords', 'cast', 'director', 
                        'original_language', 'origin_country', 'content_type']
        
        # Mevcut sütunları kontrol et
        available_columns = [col for col in text_columns if col in row.index]
        
        if not available_columns:
            logger.warning("Hiçbir metin sütunu bulunamadı!")
            return ""
        
        # TMDB detaylarını kullanmak için kontrol et
        has_tmdb_details = 'tmdb_details' in row.index
        
        # Her sütunu uygun bir şekilde hazırla ve birleştir
        texts = []
        
        for col in available_columns:
            value = row[col]
            
            # Liste tipinde değerleri işle
            if isinstance(value, list):
                value = ' '.join(str(v) for v in value)
            
            # TMDB veri zenginleştirmesi
            if col == 'keywords' and isinstance(value, dict) and 'keywords' in value:
                keyword_items = value['keywords']['keywords']
                if isinstance(keyword_items, list):
                    texts.append(' '.join([item.get('name', '') for item in keyword_items if isinstance(item, dict)]))
            elif col == 'genres' and isinstance(value, list):
                texts.append(' '.join([genre.get('name', '') for genre in value if isinstance(genre, dict)]))
            elif col == 'cast' and isinstance(value, list):
                texts.append(' '.join([cast.get('name', '') for cast in value[:10] if isinstance(cast, dict)]))
            elif col == 'director' and isinstance(value, list):
                directors = [crew.get('name', '') for crew in value if isinstance(crew, dict) and crew.get('job') == 'Director']
                if directors:
                    texts.append(' '.join(directors))
            
            # Metni ağırlık faktörü kadar tekrarla
            repeated_text = ' '.join([str(value)] * 2)
            texts.append(repeated_text)
        
        # Tüm metinleri temizle ve birleştir
        cleaned_texts = []
        for text in texts:
            # Alfanumerik olmayan karakterleri boşlukla değiştir
            cleaned = re.sub(r'[^\w\s]', ' ', str(text))
            # Fazla boşlukları temizle
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            cleaned_texts.append(cleaned)
        
        return ' '.join(cleaned_texts)
    
    def _create_user_profiles(self, ratings_df):
        """
        Kullanıcı tercihlerine dayalı profil vektörleri oluşturur
        
        Args:
            ratings_df (pd.DataFrame): Kullanıcı değerlendirmeleri
        """
        logger.info("Kullanıcı profilleri oluşturuluyor")
        
        # Kaç kullanıcı için profil oluşturulduğunu say
        processed_users = 0
        
        # Her kullanıcı için
        for user_id, group in ratings_df.groupby('user_id'):
            # Beğenilen öğeleri filtrele (min_rating üzerinde puan verilen)
            liked_items = group[group['rating'] >= self.min_rating]
            
            if len(liked_items) == 0:
                continue
                
            # Kullanıcının profil vektörünü hesapla
            user_profile = np.zeros(self.item_features.shape[1])
            for _, row in liked_items.iterrows():
                item_id = row['item_id']
                rating = row['rating']
                
                # İçerik modelimizde bu öğe var mı?
                if item_id in self.item_map:
                    item_idx = self.item_map[item_id]
                    # Puanla ağırlıklandırılmış öğe vektörünü ekle
                    item_vector = self.item_features[item_idx].toarray().flatten()
                    user_profile += item_vector * rating
            
            # Normalize et
            if np.linalg.norm(user_profile) > 0:
                user_profile = user_profile / np.linalg.norm(user_profile)
            
            # Kullanıcı profilini kaydet
            self.user_profiles[user_id] = user_profile
            processed_users += 1
            
        logger.info(f"{processed_users} kullanıcı için profil oluşturuldu")
    
    def get_similar_items(self, item_id, n=10):
        """
        Verilen öğeye en benzer n öğeyi döndürür
        
        Args:
            item_id (int): Öğe kimliği
            n (int): Döndürülecek benzer öğe sayısı
            
        Returns:
            list: [(öğe_id, benzerlik_skoru), ...] formatında sıralanmış liste
        """
        # ID hem item_id hem de id olarak kabul edilebilir
        if item_id not in self.item_indices:
            # Uyumluluk için item_id/id çevirisi dene
            if 'id' in self.items_df.columns and 'item_id' in self.items_df.columns:
                # id değerine karşılık gelen satırı bul
                if any(self.items_df['id'] == item_id):
                    # bu id'ye karşılık gelen item_id'yi bul
                    item_rows = self.items_df[self.items_df['id'] == item_id]
                    if not item_rows.empty:
                        corresponding_item_id = item_rows.iloc[0]['item_id']
                        if corresponding_item_id in self.item_indices:
                            item_id = corresponding_item_id
                
                # item_id değerine karşılık gelen satırı bul
                elif any(self.items_df['item_id'] == item_id):
                    # bu item_id'ye karşılık gelen id'yi bul
                    item_rows = self.items_df[self.items_df['item_id'] == item_id]
                    if not item_rows.empty:
                        corresponding_id = item_rows.iloc[0]['id']
                        if corresponding_id in self.item_indices:
                            item_id = corresponding_id
        
        # Eğer hala bulunamadıysa boş liste döndür
        if item_id not in self.item_indices:
            print(f"ID: {item_id} için model indeksi bulunamadı")
            return []
            
        # Öğenin dizinini al
        idx = self.item_indices[item_id]
        
        # Benzerlik skorlarını al
        if self.cosine_sim is None or idx >= len(self.cosine_sim):
            print(f"Cosine similarity matrisi yetersiz. Matris boyutu: {self.cosine_sim.shape if self.cosine_sim is not None else 'Yok'}, İstenen indeks: {idx}")
            return []
            
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Kendisi hariç benzerlik skorlarına göre sırala
        sim_scores = [(i, score) for i, score in sim_scores if i != idx]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # En benzer n öğeyi al
        sim_scores = sim_scores[:n]
        
        # Benzerlik limitini düşürelim - daha fazla uyumlu öğe bulsun
        # sim_scores = [(i, score) for i, score in sim_scores if score > 0.01]
        
        # (öğe_id, benzerlik_skoru) formatına dönüştür
        id_column = 'id' if 'id' in self.items_df.columns else 'item_id'
        similar_items = []
        for i, score in sim_scores:
            if i < len(self.items_df):  # Sınırları kontrol et
                item_id = self.items_df.iloc[i][id_column]
                if pd.notna(item_id):
                    similar_items.append((item_id, float(score)))
        
        return similar_items
    
    def recommend_for_user(self, user_id, n=10, exclude_watched=True, ratings_df=None):
        """
        Kullanıcı için önerileri döndürür
        
        Args:
            user_id: Kullanıcı ID
            n (int): Önerilecek öğe sayısı
            exclude_watched (bool): İzlenmiş öğeleri hariç tut
            ratings_df (pd.DataFrame): Kullanıcı değerlendirmeleri
            
        Returns:
            list: (item_id, score) ikililerinden oluşan liste
        """
        # Kullanıcı profili yoksa oluştur
        if user_id not in self.user_profiles and ratings_df is not None:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            if len(user_ratings) > 0:
                self._create_user_profile_for_single_user(user_id, ratings_df)
        
        # Kullanıcı profili hala yoksa, benzer kullanıcılar üzerinden öneri yap
        if user_id not in self.user_profiles:
            logger.warning(f"Kullanıcı profili bulunamadı: {user_id}")
            
            # Hiç değerlendirme verimiz yoksa popüler içerikleri öner
            if ratings_df is None or len(ratings_df) == 0:
                logger.info("Değerlendirme verisi yok, popüler içerikler öneriliyor")
                # Rastgele 10 içerik seç
                indices = np.random.choice(len(self.item_ids), min(n, len(self.item_ids)), replace=False)
                return [(self.rev_item_map[idx], 0.5) for idx in indices]
            
            # Değerlendirme verisi varsa, benzer kullanıcılar üzerinden öneri yap
            similar_user_id = self._find_similar_user(user_id, ratings_df)
            
            if similar_user_id is not None and similar_user_id in self.user_profiles:
                logger.info(f"Benzer kullanıcı bulundu: {similar_user_id}")
                user_id = similar_user_id
            else:
                # Benzer kullanıcı bulunamadıysa, popüler içerikleri öner
                logger.info("Benzer kullanıcı bulunamadı, popüler içerikler öneriliyor")
                # Rastgele 10 içerik seç
                indices = np.random.choice(len(self.item_ids), min(n, len(self.item_ids)), replace=False)
                return [(self.rev_item_map[idx], 0.5) for idx in indices]
        
        # İzlenen içerikleri belirle
        watched = set()
        if exclude_watched and ratings_df is not None:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            watched = set(user_ratings['item_id'].tolist())
        
        # Kullanıcı profili ile tüm öğeler arasındaki benzerliği hesapla
        user_profile = self.user_profiles[user_id]
        user_profile = user_profile.reshape(1, -1)
        
        # İçerik-kullanıcı benzerliği
        similarity_scores = cosine_similarity(user_profile, self.item_features).flatten()
        
        # Her öğe için puan ve ID
        item_scores = []
        for idx, score in enumerate(similarity_scores):
            # Geçerli bir anahtarsa ekle
            if idx in self.rev_item_map:
                item_id = self.rev_item_map[idx]
                item_scores.append((item_id, score))
        
        # İzlenen içerikleri filtrele
        if exclude_watched:
            item_scores = [(item_id, score) for item_id, score in item_scores if item_id not in watched]
        
        # En yüksek puanlara göre sırala
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:n]
    
    def _create_user_profile_for_single_user(self, user_id, user_ratings_df):
        """
        Tek bir kullanıcı için profil vektörü oluşturur
        
        Args:
            user_id: Kullanıcı ID
            user_ratings_df (pd.DataFrame): Kullanıcının değerlendirmeleri
        """
        logger.info(f"Kullanıcı {user_id} için profil vektörü oluşturuluyor")
        
        # Beğenilen öğeleri filtrele (min_rating üzerinde puan verilen)
        liked_items = user_ratings_df[user_ratings_df['rating'] >= self.min_rating]
        
        if len(liked_items) == 0:
            logger.warning(f"Kullanıcı {user_id}'nin beğendiği hiçbir içerik bulunamadı")
            return
            
        # Kullanıcının profil vektörünü hesapla
        user_profile = np.zeros(self.item_features.shape[1])
        profile_items_count = 0
        
        for _, row in liked_items.iterrows():
            item_id = row['item_id']
            rating = row['rating']
            
            # İçerik modelimizde bu öğe var mı?
            if item_id in self.item_map:
                item_idx = self.item_map[item_id]
                # Puanla ağırlıklandırılmış öğe vektörünü ekle
                item_vector = self.item_features[item_idx].toarray().flatten()
                user_profile += item_vector * rating
                profile_items_count += 1
        
        # En az bir içerik işlendiyse profili kaydet
        if profile_items_count > 0:
            # Normalize et
            if np.linalg.norm(user_profile) > 0:
                user_profile = user_profile / np.linalg.norm(user_profile)
            
            # Kullanıcı profilini kaydet
            self.user_profiles[user_id] = user_profile
            logger.info(f"Kullanıcı {user_id} için profil oluşturuldu ({profile_items_count} içerik)")
        else:
            logger.warning(f"Kullanıcı {user_id} için profil oluşturulamadı - hiçbir içerik vektörü bulunamadı")
    
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
            ContentBasedFiltering: Yüklenen model
        """
        if os.path.exists(filepath):
            logger.info(f"Model yükleniyor: {filepath}")
            return joblib.load(filepath)
        else:
            logger.warning(f"Model bulunamadı: {filepath}")
            return None 