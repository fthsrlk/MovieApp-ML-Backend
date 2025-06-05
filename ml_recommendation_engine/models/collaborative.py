"""
İşbirlikçi filtreleme (Collaborative Filtering) algoritması
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    """
    İşbirlikçi filtreleme modeli
    
    Kullanıcı-öğe matrisinde kullanıcıların benzerliği temelinde öneri yapar
    """
    
    def __init__(self, method='user-based', num_factors=100, reg_param=0.1):
        """
        Args:
            method (str): 'user-based' veya 'item-based' veya 'matrix-factorization'
            num_factors (int): Matris faktörizasyonu için gizli faktör sayısı
            reg_param (float): Düzenlileştirme parametresi
        """
        self.method = method
        self.num_factors = num_factors
        self.reg_param = reg_param
        self.model = None
        self.user_item_matrix = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.item_similarity = None
        self.user_similarity = None
        self.user_map = {}  # user_id -> index
        self.item_map = {}  # item_id -> index
        self.rev_user_map = {}  # index -> user_id
        self.rev_item_map = {}  # index -> item_id
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, ratings_df):
        """
        Modeli eğit
        
        Args:
            ratings_df (pd.DataFrame): user_id, item_id, rating sütunlarını içeren
                değerlendirme verileri
        """
        logger.info(f"CollaborativeFiltering.fit başladı, method={self.method}")
        
        # Aynı kullanıcı-öğe ikilisinin birden fazla değerlendirmesi varsa, ortalama al
        if not ratings_df.empty:
            ratings_df = ratings_df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()
        
        # Eşleme oluştur
        self.user_map = {user: i for i, user in enumerate(ratings_df['user_id'].unique())}
        self.item_map = {item: i for i, item in enumerate(ratings_df['item_id'].unique())}
        self.rev_user_map = {i: user for user, i in self.user_map.items()}
        self.rev_item_map = {i: item for item, i in self.item_map.items()}
        
        # Kullanıcı-öğe matrisi oluştur
        matrix_df = ratings_df.copy()
        matrix_df['user_idx'] = matrix_df['user_id'].map(self.user_map)
        matrix_df['item_idx'] = matrix_df['item_id'].map(self.item_map)
        
        # Pivot işleminde index ve columns değerlerinin benzersiz olduğundan emin ol
        pivot_df = matrix_df.pivot(
            index='user_idx', 
            columns='item_idx', 
            values='rating'
        ).fillna(0)
        
        # Pivot matrisini numpy dizisine dönüştür
        self.user_item_matrix = pivot_df.to_numpy()
        
        # Global ortalama
        self.global_mean = ratings_df['rating'].mean()
        
        if self.method == 'user-based':
            self._fit_user_based()
        elif self.method == 'item-based':
            self._fit_item_based()
        elif self.method == 'matrix-factorization':
            self._fit_matrix_factorization()
        else:
            raise ValueError(f"Bilinmeyen yöntem: {self.method}")
        
        logger.info(f"CollaborativeFiltering.fit tamamlandı, method={self.method}")
        return self
    
    def _fit_user_based(self):
        """
        Kullanıcı tabanlı işbirlikçi filtreleme modeli eğitimi
        """
        # Transpoze et çünkü sklearn cosine_similarity satırlar arasında hesaplama yapar
        user_matrix = self.user_item_matrix
        # Kullanıcı-kullanıcı benzerlik matrisi
        self.user_similarity = cosine_similarity(user_matrix)
        
    def _fit_item_based(self):
        """
        Öğe tabanlı işbirlikçi filtreleme modeli eğitimi
        """
        item_matrix = self.user_item_matrix.T
        # Öğe-öğe benzerlik matrisi
        self.item_similarity = cosine_similarity(item_matrix)
    
    def _fit_matrix_factorization(self):
        """
        Matris faktörizasyonu yaklaşımı ile eğitim
        """
        # Numpy matrisine dönüştür
        matrix = self.user_item_matrix
        
        # Kullanıcı ve öğe ortalamaları (bias değerleri)
        user_rated_items = (matrix > 0).astype(float)  # Kullanıcının değerlendirdiği öğeler
        n_ratings_per_user = np.sum(user_rated_items, axis=1)
        n_ratings_per_item = np.sum(user_rated_items, axis=0)
        
        # Bias hesapla (daha dengeli bir yaklaşım)
        self.global_mean = np.sum(matrix) / np.sum(user_rated_items) if np.sum(user_rated_items) > 0 else 0.0
        
        # Kullanıcı bias hesaplama (sıfıra bölünmeyi engelle)
        self.user_bias = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            if n_ratings_per_user[i] > 0:
                self.user_bias[i] = np.sum(matrix[i, :]) / n_ratings_per_user[i] - self.global_mean
        
        # Öğe bias hesaplama (sıfıra bölünmeyi engelle)
        self.item_bias = np.zeros(matrix.shape[1])
        for i in range(matrix.shape[1]):
            if n_ratings_per_item[i] > 0:
                self.item_bias[i] = np.sum(matrix[:, i]) / n_ratings_per_item[i] - self.global_mean
        
        # Bias'ları çıkararak normalize edilmiş matris oluştur
        normalized_matrix = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0:  # Değerlendirme varsa
                    normalized_matrix[i, j] = matrix[i, j] - self.global_mean - self.user_bias[i] - self.item_bias[j]
        
        # SVD uygula - faktör sayısı en az 1 olmalı
        factors = min(self.num_factors, min(matrix.shape) - 1)
        if factors < 1:
            factors = 1
        
        # SVD uygula
        try:
            # Eğer matris boşsa veya tek değere sahipse SVD başarısız olabilir
            if np.count_nonzero(normalized_matrix) <= 1:
                raise Exception("Yetersiz veri: Matris çoğunlukla sıfır veya tek değere sahip")
                
            # SVD ile faktörizasyon
            U, sigma, Vt = svds(normalized_matrix, k=factors)
            
            # Düzenlileştirme parametresi uygula
            if self.reg_param > 0:
                # L2 düzenlileştirme
                sigma_reg = sigma / (1 + self.reg_param)
            else:
                sigma_reg = sigma
                
            # Sigma değerlerini doğrudan faktör matrislerine dahil et
            sigma_diag = np.diag(np.sqrt(sigma_reg))
            
            # Kullanıcı ve öğe faktörleri
            self.user_factors = np.dot(U, sigma_diag)
            self.item_factors = np.dot(Vt.T, sigma_diag)
            
            logger.info(f"Matris faktörizasyonu tamamlandı: {self.user_factors.shape}, {self.item_factors.shape}")
        except Exception as e:
            logger.error(f"SVD hatası: {str(e)}")
            # Hata durumunda basit bir yaklaşım kullan
            self.user_factors = np.random.rand(matrix.shape[0], factors) * 0.1
            self.item_factors = np.random.rand(matrix.shape[1], factors) * 0.1
            logger.info("SVD hatası nedeniyle rastgele faktörler üretildi")
    
    def predict(self, user_id, item_id):
        """
        Belirli bir kullanıcı ve öğe için puan tahmini yapar
        
        Args:
            user_id: Kullanıcı ID
            item_id: Öğe ID
            
        Returns:
            float: Tahmin edilen puan
        """
        if user_id not in self.user_map or item_id not in self.item_map:
            return self.global_mean
        
        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]
        
        if self.method == 'user-based':
            return self._predict_user_based(user_idx, item_idx)
        elif self.method == 'item-based':
            return self._predict_item_based(user_idx, item_idx)
        elif self.method == 'matrix-factorization':
            return self._predict_matrix_factorization(user_idx, item_idx)
        else:
            raise ValueError(f"Bilinmeyen yöntem: {self.method}")
    
    def _predict_user_based(self, user_idx, item_idx):
        """
        Kullanıcı tabanlı CF tahmini
        """
        # Diğer kullanıcıların bu öğeye verdiği puanları al
        item_ratings = self.user_item_matrix[:, item_idx]
        
        # Mevcut kullanıcının diğer kullanıcılara olan benzerliği
        similarities = self.user_similarity[user_idx, :]
        
        # Benzerlikler sıfırsa, global ortalamayı döndür
        if np.sum(similarities) == 0:
            return self.global_mean
        
        # Ağırlıklı ortalama hesapla
        weighted_sum = np.sum(similarities * item_ratings)
        similarity_sum = np.sum(np.abs(similarities))
        
        return weighted_sum / similarity_sum if similarity_sum > 0 else self.global_mean
    
    def _predict_item_based(self, user_idx, item_idx):
        """
        Öğe tabanlı CF tahmini
        """
        # Kullanıcının değerlendirdiği diğer öğeleri al
        user_ratings = self.user_item_matrix[user_idx, :]
        
        # Mevcut öğenin diğer öğelere olan benzerliği
        similarities = self.item_similarity[item_idx, :]
        
        # 0 olmayan değerlendirmeleri filtrele
        mask = user_ratings != 0
        if not np.any(mask):
            return self.global_mean
        
        # Sadece kullanıcının değerlendirdiği öğeleri dikkate al
        relevant_similarities = similarities[mask]
        relevant_ratings = user_ratings[mask]
        
        # Benzerlikler sıfırsa, global ortalamayı döndür
        if np.sum(relevant_similarities) == 0:
            return self.global_mean
        
        # Ağırlıklı ortalama hesapla
        weighted_sum = np.sum(relevant_similarities * relevant_ratings)
        similarity_sum = np.sum(np.abs(relevant_similarities))
        
        return weighted_sum / similarity_sum if similarity_sum > 0 else self.global_mean
    
    def _predict_matrix_factorization(self, user_idx, item_idx):
        """
        Matris faktörizasyonu tahmini
        """
        if self.user_factors is None or self.item_factors is None:
            return self.global_mean
            
        # Kullanıcı x Öğe tahmini
        try:
            # Faktör vektörleri arasındaki dot product
            mf_prediction = np.dot(
                self.user_factors[user_idx, :], 
                self.item_factors[item_idx, :]
            )
            
            # Bias değerleri ekle
            prediction = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx] + mf_prediction
            
            # Tahmini makul bir aralığa sınırla (1-5 arası gibi)
            min_rating = 1.0
            max_rating = 5.0
            prediction = max(min_rating, min(max_rating, prediction))
            
            return prediction
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            return self.global_mean
    
    def recommend(self, user_id, n=10, exclude_watched=True):
        """
        Kullanıcı için önerileri döndürür
        
        Args:
            user_id: Kullanıcı ID
            n (int): Önerilecek öğe sayısı
            exclude_watched (bool): İzlenmiş öğeleri hariç tut
            
        Returns:
            list: (item_id, predicted_rating) ikililerinden oluşan liste
        """
        if user_id not in self.user_map:
            logger.warning(f"Kullanıcı bulunamadı: {user_id}")
            # Kullanıcı yoksa, popüler öğeler arasından rastgele öneriler döndür
            # Popülerlik için öğe bias'ı kullan
            if self.item_bias is not None and len(self.item_bias) > 0:
                popular_items = [(self.rev_item_map[i], score) for i, score in enumerate(self.item_bias)]
                popular_items.sort(key=lambda x: x[1], reverse=True)
                top_n = min(n*3, len(popular_items))
                if top_n > 0:
                    import random
                    return random.sample(popular_items[:top_n], min(n, top_n))
            
            # Bias yoksa rastgele seç
            random_items = list(self.item_map.keys())
            np.random.shuffle(random_items)
            return [(item_id, 0.5) for item_id in random_items[:n]]
        
        user_idx = self.user_map[user_id]
        
        # Kullanıcının izledikleri
        watched = set()
        if exclude_watched:
            watched_indices = np.where(self.user_item_matrix[user_idx, :] > 0)[0]
            watched = {self.rev_item_map[idx] for idx in watched_indices}
        
        # Matris faktörizasyonu için daha verimli hesaplama
        if self.method == 'matrix-factorization' and self.user_factors is not None and self.item_factors is not None:
            try:
                # Kullanıcının faktör vektörü
                user_vector = self.user_factors[user_idx, :]
                
                # Tüm öğelerle benzerlik hesapla
                item_scores = np.dot(self.item_factors, user_vector)
                
                # Bias ekle
                item_scores = item_scores + self.global_mean + self.user_bias[user_idx] + self.item_bias
                
                # Değerlendirmeleri ayarla
                min_rating = 1.0
                max_rating = 5.0
                item_scores = np.clip(item_scores, min_rating, max_rating)
                
                # En yüksek puanlı öğeleri seç
                top_indices = np.argsort(item_scores)[::-1]
                
                # İzlenen öğeleri filtrele
                if exclude_watched:
                    top_indices = [i for i in top_indices if self.rev_item_map[i] not in watched]
                
                # İlk n öğeyi al
                predictions = [(self.rev_item_map[i], item_scores[i]) for i in top_indices[:n]]
                return predictions
            except Exception as e:
                logger.error(f"Verimli öneri hatası: {str(e)}, yedek yönteme geçiliyor")
                # Hata durumunda yedek yönteme geç
        
        # Geleneksel yöntem (tüm öğeler için tek tek tahmin)
        predictions = []
        for item_id, item_idx in self.item_map.items():
            if item_id in watched:
                continue
            
            # Tahmini hesapla
            predicted_rating = self.predict(user_id, item_id)
            predictions.append((item_id, predicted_rating))
        
        # En yüksek tahminlere göre sırala
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
    
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
            CollaborativeFiltering: Yüklenen model
        """
        if os.path.exists(filepath):
            logger.info(f"Model yükleniyor: {filepath}")
            return joblib.load(filepath)
        else:
            logger.warning(f"Model bulunamadı: {filepath}")
            return None 