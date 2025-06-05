"""
Veri ön işleme modülü
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Öneri sistemi için veri ön işleme
    """
    
    def __init__(self):
        """
        Veri ön işleme sınıfını başlat
        """
        self.scaler = MinMaxScaler()
    
    def preprocess_items(self, items_df):
        """
        İçerik verilerini ön işle
        
        Args:
            items_df (pd.DataFrame): İçerik verileri
            
        Returns:
            pd.DataFrame: İşlenmiş veriler
        """
        logger.info("İçerik verileri ön işleniyor")
        
        # Kopya oluştur
        df = items_df.copy()
        
        # Eksik değerleri doldur
        df['overview'] = df['overview'].fillna('')
        
        # Tarih alanlarını işle
        self._process_dates(df)
        
        # Liste tipindeki alanları işle
        self._process_list_fields(df)
        
        # Sayısal alanları normalize et
        self._normalize_numeric_fields(df)
        
        # Özellik metni oluştur
        df = self._create_feature_text(df)
        
        logger.info("İçerik verileri ön işleme tamamlandı")
        return df
    
    def preprocess_ratings(self, ratings_df, min_ratings=5):
        """
        Değerlendirme verilerini ön işle
        
        Args:
            ratings_df (pd.DataFrame): Değerlendirme verileri
            min_ratings (int): Bir kullanıcının minimum değerlendirme sayısı
            
        Returns:
            pd.DataFrame: İşlenmiş veriler
        """
        logger.info("Değerlendirme verileri ön işleniyor")
        
        # Kopya oluştur
        df = ratings_df.copy()
        
        # Eksik değerleri temizle
        df = df.dropna(subset=['user_id', 'item_id', 'rating'])
        
        # Veri tiplerini düzelt
        df['user_id'] = df['user_id'].astype(int)
        df['item_id'] = df['item_id'].astype(int)
        df['rating'] = df['rating'].astype(float)
        
        # Timestamp varsa işle
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['timestamp'] = datetime.now()
        
        # Az değerlendirme yapan kullanıcıları filtrele
        if min_ratings > 0:
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_ratings].index
            df = df[df['user_id'].isin(valid_users)]
        
        logger.info(f"Değerlendirme verileri ön işleme tamamlandı: {len(df)} değerlendirme, {df['user_id'].nunique()} kullanıcı")
        return df
    
    def _process_dates(self, df):
        """
        Tarih alanlarını işle
        
        Args:
            df (pd.DataFrame): İşlenecek veri
        """
        # Film tarihleri
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['release_year'] = df['release_date'].dt.year
        
        # Dizi tarihleri
        if 'first_air_date' in df.columns:
            df['first_air_date'] = pd.to_datetime(df['first_air_date'], errors='coerce')
            df['first_air_year'] = df['first_air_date'].dt.year
        
        if 'last_air_date' in df.columns:
            df['last_air_date'] = pd.to_datetime(df['last_air_date'], errors='coerce')
    
    def _process_list_fields(self, df):
        """
        Liste tipindeki alanları işle
        
        Args:
            df (pd.DataFrame): İşlenecek veri
        """
        # Türleri işle
        if 'genres' in df.columns:
            # eval() kullanmak yerine daha güvenli bir yöntem kullan
            def parse_list(value):
                if pd.isna(value) or value is None:
                    return []
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    try:
                        import ast
                        return ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        # Virgülle ayrılmış değerler olabilir
                        return [item.strip() for item in value.split(',') if item.strip()]
                return []
                
            df['genres'] = df['genres'].apply(parse_list)
        
        # Oyuncuları işle
        if 'cast' in df.columns:
            df['cast'] = df['cast'].apply(parse_list)
        
        # Anahtar kelimeleri işle
        if 'keywords' in df.columns:
            df['keywords'] = df['keywords'].apply(parse_list)
    
    def _normalize_numeric_fields(self, df):
        """
        Sayısal alanları normalize et
        
        Args:
            df (pd.DataFrame): İşlenecek veri
        """
        numeric_cols = ['popularity', 'vote_average', 'vote_count']
        
        for col in numeric_cols:
            if col in df.columns:
                # Eksik değerleri doldur
                df[col] = df[col].fillna(0)
                
                # Aykırı değerleri kırp
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                df[col] = df[col].clip(upper=upper_bound)
                
                # Normalize et
                df[f'{col}_norm'] = self.scaler.fit_transform(df[[col]])
    
    def _create_feature_text(self, df):
        """
        Özellik metni oluştur
        
        Args:
            df (pd.DataFrame): İşlenecek veri
            
        Returns:
            pd.DataFrame: Özellik metni eklenmiş veri
        """
        # Kullanılabilecek sütunlar
        text_columns = ['title', 'overview', 'genres', 'keywords', 'cast', 'director']
        
        # Mevcut sütunları kontrol et
        available_columns = [col for col in text_columns if col in df.columns]
        
        if not available_columns:
            logger.warning("Hiçbir metin sütunu bulunamadı!")
            df['features_text'] = ""
            return df
        
        # Her sütunu uygun bir şekilde hazırla ve birleştir
        features = []
        
        for _, row in df.iterrows():
            # Her sütunu dizeye dönüştür ve birleştir
            texts = []
            
            for col in available_columns:
                value = row.get(col, '')
                
                # Liste tipinde değerleri işle
                if isinstance(value, list):
                    value = ' '.join(str(v) for v in value)
                
                # None olmadığından emin ol
                if value is not None:
                    # Önemli sütunları tekrarla (ağırlıklandırma)
                    if col in ['title', 'genres', 'keywords']:
                        texts.append(str(value) + ' ' + str(value))
                    else:
                        texts.append(str(value))
            
            # Tüm metinleri birleştir
            feature_text = ' '.join(texts)
            features.append(feature_text)
        
        # Yeni sütunu ekle
        df['features_text'] = features
        
        return df
    
    def split_train_test(self, ratings_df, test_size=0.2, random_state=42):
        """
        Değerlendirme verilerini eğitim ve test kümelerine ayır
        
        Args:
            ratings_df (pd.DataFrame): Değerlendirme verileri
            test_size (float): Test kümesi oranı
            random_state (int): Rastgele durum
            
        Returns:
            tuple: (train_df, test_df)
        """
        # Her kullanıcı için ayrı ayrı böl
        train_data = []
        test_data = []
        
        for user_id, group in ratings_df.groupby('user_id'):
            # Rastgele karıştır
            group = group.sample(frac=1, random_state=random_state)
            
            # Böl
            n_test = max(1, int(len(group) * test_size))
            user_test = group.iloc[:n_test]
            user_train = group.iloc[n_test:]
            
            train_data.append(user_train)
            test_data.append(user_test)
        
        # Birleştir
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        logger.info(f"Veri bölme tamamlandı: {len(train_df)} eğitim, {len(test_df)} test")
        return train_df, test_df 