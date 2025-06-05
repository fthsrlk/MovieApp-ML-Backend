"""
Veri yükleme modülü
"""

import pandas as pd
import numpy as np
import requests
import os
import logging
import json
from datetime import datetime
import re
import math
import time

logger = logging.getLogger(__name__)

class TMDBDataLoader:
    """
    TMDB API'sinden veri yükleme ve işleme
    """
    
    def __init__(self, api_key, language='tr-TR'):
        """
        Args:
            api_key (str): TMDB API anahtarı
            language (str): İçerik dili (örn. 'tr-TR', 'en-US')
        """
        self.api_key = api_key
        self.language = language
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p"
        self.cache_dir = "cache"
        
        # Cache dizinini oluştur
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_movie_details(self, movie_id, use_cache=True):
        """
        Film detaylarını al
        
        Args:
            movie_id (int): TMDB film ID
            use_cache (bool): Önbellek kullan
            
        Returns:
            dict: Film detayları
        """
        cache_file = os.path.join(self.cache_dir, f"movie_{movie_id}.json")
        
        # Önbellekten yükle
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # API'den al
        url = f"{self.base_url}/movie/{movie_id}"
        params = {
            'api_key': self.api_key,
            'language': self.language,
            'append_to_response': 'credits,keywords,videos,recommendations'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Önbelleğe kaydet
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return data
        else:
            logger.error(f"Film detayları alınamadı: {movie_id}, Hata: {response.status_code}")
            return None
    
    def get_tv_details(self, tv_id, use_cache=True):
        """
        Dizi detaylarını al
        
        Args:
            tv_id (int): TMDB dizi ID
            use_cache (bool): Önbellek kullan
            
        Returns:
            dict: Dizi detayları
        """
        cache_file = os.path.join(self.cache_dir, f"tv_{tv_id}.json")
        
        # Önbellekten yükle
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # API'den al
        url = f"{self.base_url}/tv/{tv_id}"
        params = {
            'api_key': self.api_key,
            'language': self.language,
            'append_to_response': 'credits,keywords,videos,recommendations'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Önbelleğe kaydet
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return data
        else:
            logger.error(f"Dizi detayları alınamadı: {tv_id}, Hata: {response.status_code}")
            return None
    
    def search_movies(self, query, page=1):
        """
        Film ara
        
        Args:
            query (str): Arama sorgusu
            page (int): Sayfa numarası
            
        Returns:
            dict: Arama sonuçları
        """
        url = f"{self.base_url}/search/movie"
        params = {
            'api_key': self.api_key,
            'language': self.language,
            'query': query,
            'page': page,
            'include_adult': 'false'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Film araması başarısız: {query}, Hata: {response.status_code}")
            return {'results': []}
    
    def search_tv(self, query, page=1):
        """
        Dizi ara
        
        Args:
            query (str): Arama sorgusu
            page (int): Sayfa numarası
            
        Returns:
            dict: Arama sonuçları
        """
        url = f"{self.base_url}/search/tv"
        params = {
            'api_key': self.api_key,
            'language': self.language,
            'query': query,
            'page': page,
            'include_adult': 'false'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Dizi araması başarısız: {query}, Hata: {response.status_code}")
            return {'results': []}
    
    def get_popular_movies(self, page=1, limit=None):
        """
        TMDB'den popüler filmleri alır
        
        Args:
            page (int): Sayfa numarası
            limit (int, optional): Maksimum film sayısı
            
        Returns:
            list: Film listesi
        """
        max_retries = 3
        retry_delay = 2  # saniye
        
        url = f"{self.base_url}/movie/popular"
        params = {
            'api_key': self.api_key,
            'language': self.language,
            'page': page
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status() # HTTP hataları için exception fırlat

                data = response.json()
                results = data.get('results', [])
                
                if limit is not None:
                    results = results[:limit]
                
                return results
            
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"Popüler filmler için HTTP hatası (Deneme: {attempt+1}/{max_retries}): {http_err} - URL: {url}")
                if response.status_code == 429: # Rate limit
                    logger.warning("Rate limit aşıldı, daha uzun süre bekleniyor...")
                    time.sleep(retry_delay * 2) 
                    retry_delay *= 2 
                elif attempt == max_retries - 1:
                    logger.error(f"Popüler filmler için tüm HTTP denemeleri başarısız oldu: {url}")
                    return [] # Son denemede HTTP hatası alınırsa boş liste dön
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Popüler filmler için istek hatası (Deneme: {attempt+1}/{max_retries}): {req_err} - URL: {url}")
            
            if attempt < max_retries - 1:
                logger.info(f"Popüler filmler için yeniden deneniyor ({attempt+2}/{max_retries}), {retry_delay} saniye sonra...")
                time.sleep(retry_delay)
            else: # Son denemeden sonra hala başarılı olunamadıysa
                logger.error(f"Popüler filmler alınamadı, {max_retries} deneme başarısız oldu.")
                return [] # Tüm denemeler bittikten sonra boş liste dön
        
        return [] # Döngü normal şekilde biterse (hiç deneme yapılmazsa gibi), yine de boş liste dön

    def get_popular_tv_shows(self, page=1, limit=None):
        """
        TMDB'den popüler dizileri alır
        
        Args:
            page (int): Sayfa numarası
            limit (int, optional): Maksimum dizi sayısı
            
        Returns:
            list: Dizi listesi
        """
        max_retries = 3
        retry_delay = 2  # saniye
        
        url = f"{self.base_url}/tv/popular"
        params = {
            'api_key': self.api_key,
            'language': self.language,
            'page': page
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status() # HTTP hataları için exception fırlat

                data = response.json()
                results = data.get('results', [])
                
                if limit is not None:
                    results = results[:limit]
                    
                return results

            except requests.exceptions.HTTPError as http_err:
                logger.error(f"Popüler diziler için HTTP hatası (Deneme: {attempt+1}/{max_retries}): {http_err} - URL: {url}")
                if response.status_code == 429: # Rate limit
                    logger.warning("Rate limit aşıldı, daha uzun süre bekleniyor...")
                    time.sleep(retry_delay * 2)
                    retry_delay *= 2 
                elif attempt == max_retries - 1:
                    logger.error(f"Popüler diziler için tüm HTTP denemeleri başarısız oldu: {url}")
                    return [] 
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Popüler diziler için istek hatası (Deneme: {attempt+1}/{max_retries}): {req_err} - URL: {url}")

            if attempt < max_retries - 1:
                logger.info(f"Popüler diziler için yeniden deneniyor ({attempt+2}/{max_retries}), {retry_delay} saniye sonra...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Popüler diziler alınamadı, {max_retries} deneme başarısız oldu.")
                return []
        
        return []
    
    def get_movie_genres(self):
        """
        Film türlerini al
        
        Returns:
            list: Tür listesi
        """
        url = f"{self.base_url}/genre/movie/list"
        params = {
            'api_key': self.api_key,
            'language': self.language
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()['genres']
        else:
            logger.error(f"Film türleri alınamadı, Hata: {response.status_code}")
            return []
    
    def get_tv_genres(self):
        """
        Dizi türlerini al
        
        Returns:
            list: Tür listesi
        """
        url = f"{self.base_url}/genre/tv/list"
        params = {
            'api_key': self.api_key,
            'language': self.language
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()['genres']
        else:
            logger.error(f"Dizi türleri alınamadı, Hata: {response.status_code}")
            return []
    
    def get_image_url(self, path, size='w500'):
        """
        Görsel URL'si oluştur
        
        Args:
            path (str): Görsel yolu
            size (str): Görsel boyutu (w92, w154, w185, w342, w500, w780, original)
            
        Returns:
            str: Tam görsel URL'si
        """
        if not path:
            return None
        return f"{self.image_base_url}/{size}{path}"
    
    def convert_to_dataframe(self, items, content_type='movie'):
        """
        API sonuçlarını DataFrame'e dönüştür
        
        Args:
            items (list): API sonuçları
            content_type (str): 'movie' veya 'tv'
            
        Returns:
            pd.DataFrame: İçerik DataFrame'i
        """
        if not items:
            return pd.DataFrame()
        
        records = []
        
        for item in items:
            # Ortak alanlar
            record = {
                'item_id': item['id'],
                'content_type': content_type,
                'title': item.get('title' if content_type == 'movie' else 'name', ''),
                'overview': item.get('overview', ''),
                'poster_path': self.get_image_url(item.get('poster_path')),
                'backdrop_path': self.get_image_url(item.get('backdrop_path')),
                'popularity': item.get('popularity', 0),
                'vote_average': item.get('vote_average', 0),
                'vote_count': item.get('vote_count', 0)
            }
            
            # İçerik tipine göre özel alanlar
            if content_type == 'movie':
                record.update({
                    'release_date': item.get('release_date', ''),
                    'runtime': item.get('runtime', 0)
                })
            else:  # TV
                record.update({
                    'first_air_date': item.get('first_air_date', ''),
                    'last_air_date': item.get('last_air_date', ''),
                    'number_of_seasons': item.get('number_of_seasons', 0),
                    'number_of_episodes': item.get('number_of_episodes', 0)
                })
            
            # Türler
            if 'genres' in item:
                record['genres'] = [genre['name'] for genre in item['genres']]
            elif 'genre_ids' in item:
                # Tür ID'lerini isimlere dönüştürmek için ek işlem gerekir
                record['genre_ids'] = item['genre_ids']
            
            # Oyuncular ve ekip
            if 'credits' in item:
                if 'cast' in item['credits']:
                    record['cast'] = [person['name'] for person in item['credits']['cast'][:10]]
                
                if 'crew' in item['credits']:
                    directors = [person['name'] for person in item['credits']['crew'] 
                                if person['job'] == 'Director']
                    record['director'] = directors[0] if directors else ''
            
            # Anahtar kelimeler
            if 'keywords' in item:
                keywords_list = item['keywords'].get('keywords', []) if isinstance(item['keywords'], dict) else item['keywords']
                record['keywords'] = [keyword['name'] for keyword in keywords_list]
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _clean_string_values(self, data):
        """
        Bir veri sözlüğündeki metin değerlerini temizler
        
        Args:
            data (dict): Temizlenecek veri sözlüğü
            
        Returns:
            dict: Temizlenmiş veri sözlüğü
        """
        if not isinstance(data, dict):
            return data
            
        for key, value in data.items():
            if isinstance(value, str):
                # Özel karakterleri temizle
                value = re.sub(r'[\n\r\t\\"]', ' ', value)
                # Çift boşlukları temizle
                value = re.sub(r'\s+', ' ', value)
                # Başındaki ve sonundaki boşlukları kaldır
                data[key] = value.strip()
            elif isinstance(value, list):
                # Listedeki her öğeyi temizle
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        value[i] = self._clean_string_values(item)
            elif isinstance(value, dict):
                # İç içe sözlükleri temizle
                data[key] = self._clean_string_values(value)
                
        return data
        
    def load_sample_data(self, n_movies=300, n_tv=200, n_users=100, min_vote_count=100, seed=42):
        """
        Örnek veri kümesi yükle
        
        Args:
            n_movies (int): Film sayısı
            n_tv (int): Dizi sayısı
            n_users (int): Kullanıcı sayısı
            min_vote_count (int): Minimum oy sayısı
            seed (int): Rastgelelik tohumu
            
        Returns:
            tuple: (items_df, ratings_df)
        """
        np.random.seed(seed)
        
        # Daha güvenli veri yükleme için tasarlanmış n_movies ve n_tv değerleri
        max_movies = min(n_movies, 1000)  # Fazla film istenirse sınırla (API için)
        max_tv = min(n_tv, 500)  # Fazla dizi istenirse sınırla (API için)
        
        # Her sayfada 20 sonuç döndürüldüğünden, kaç sayfa gerektiğini hesapla
        movies_pages = max(1, math.ceil(max_movies / 20))
        tv_pages = max(1, math.ceil(max_tv / 20))
        
        logger.info(f"{max_movies} film ve {max_tv} dizi yükleniyor (sırasıyla {movies_pages} ve {tv_pages} sayfa)")
        
        # Popüler filmler
        movies = []
        for page in range(1, movies_pages + 1):
            # Son sayfada sınırlı sayıda film istenebilir
            remaining = max_movies - len(movies)
            if remaining <= 0:
                break
                
            page_limit = min(20, remaining)  # Sayfa başına en fazla 20 film
            page_results = self.get_popular_movies(page=page, limit=page_limit)
            
            if not page_results:
                logger.warning(f"Sayfa {page}'deki filmler alınamadı, devam ediliyor")
                continue
                
            movies.extend(page_results)
            
            # Yeterli film toplandıysa döngüden çık
            if len(movies) >= max_movies:
                break
                
            # API hız sınırını aşmamak için kısa bir bekleme
            time.sleep(0.25)
        
        # Popüler diziler
        tv_shows = []
        for page in range(1, tv_pages + 1):
            # Son sayfada sınırlı sayıda dizi istenebilir
            remaining = max_tv - len(tv_shows)
            if remaining <= 0:
                break
                
            page_limit = min(20, remaining)  # Sayfa başına en fazla 20 dizi
            page_results = self.get_popular_tv_shows(page=page, limit=page_limit)
            
            if not page_results:
                logger.warning(f"Sayfa {page}'deki diziler alınamadı, devam ediliyor")
                continue
                
            tv_shows.extend(page_results)
            
            # Yeterli dizi toplandıysa döngüden çık
            if len(tv_shows) >= max_tv:
                break
                
            # API hız sınırını aşmamak için kısa bir bekleme
            time.sleep(0.25)
        
        # Veriler alınamadıysa veya çok az alındıysa hata log'u yaz ve boş veriler döndür
        if not movies and not tv_shows:
            logger.error("Hiç film veya dizi alınamadı!")
            return pd.DataFrame(), pd.DataFrame()
            
        if len(movies) < max_movies * 0.1 and len(tv_shows) < max_tv * 0.1:
            logger.error(f"Çok az veri alındı: {len(movies)} film, {len(tv_shows)} dizi")
            
        # Film ve dizileri işle
        items = []
        for movie in movies:
            if movie.get('vote_count', 0) >= min_vote_count:
                items.append({
                    'item_id': movie['id'],
                    'title': movie.get('title', ''),
                    'overview': movie.get('overview', ''),
                    'poster_path': movie.get('poster_path', ''),
                    'release_date': movie.get('release_date', ''),
                    'vote_average': movie.get('vote_average', 0),
                    'vote_count': movie.get('vote_count', 0),
                    'popularity': movie.get('popularity', 0),
                    'original_language': movie.get('original_language', ''),
                    'content_type': 'movie',
                    'tmdb_details': self.get_movie_details(movie['id'])
                })
        
        for show in tv_shows:
            if show.get('vote_count', 0) >= min_vote_count:
                items.append({
                    'item_id': show['id'],
                    'title': show.get('name', ''),
                    'overview': show.get('overview', ''),
                    'poster_path': show.get('poster_path', ''),
                    'release_date': show.get('first_air_date', ''),
                    'vote_average': show.get('vote_average', 0),
                    'vote_count': show.get('vote_count', 0),
                    'popularity': show.get('popularity', 0),
                    'original_language': show.get('original_language', ''),
                    'content_type': 'tv',
                    'tmdb_details': self.get_tv_details(show['id'])
                })
        
        # Boş items_df durumunu kontrol et
        if not items:
            logger.error("Hiçbir içerik işlenemedi!")
            empty_items_df = pd.DataFrame(columns=[
                'item_id', 'title', 'overview', 'poster_path', 'release_date',
                'vote_average', 'vote_count', 'popularity', 'original_language',
                'content_type', 'tmdb_details'
            ])
            empty_ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
            return empty_items_df, empty_ratings_df
        
        items_df = pd.DataFrame(items)
        
        # Tip dönüşümleri
        items_df['vote_average'] = items_df['vote_average'].astype(float)
        items_df['vote_count'] = items_df['vote_count'].astype(int)
        
        # En az bir içerik olduğundan emin ol
        if len(items_df) == 0:
            logger.error("Hiçbir içerik işlenemedi!")
            empty_ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
            return items_df, empty_ratings_df
            
        # İçerikler için genre bilgisi çıkar
        genres = []
        for _, row in items_df.iterrows():
            # tmdb_details dictionary olarak kontrol et
            if isinstance(row['tmdb_details'], dict) and 'genres' in row['tmdb_details']:
                tmdb_genres = row['tmdb_details']['genres']
                if isinstance(tmdb_genres, list):
                    genre_names = [g['name'] for g in tmdb_genres if isinstance(g, dict) and 'name' in g]
                    genres.append(','.join(genre_names))
                else:
                    genres.append('')
            else:
                genres.append('')
        
        items_df['genres'] = genres
        
        # Kullanıcı değerlendirmeleri oluştur
        ratings = []
        timestamp = int(time.time())
        
        # Her kullanıcı en az 5 içerik değerlendirmeli, bazıları daha fazla değerlendirecek
        for user_id in range(1, n_users + 1):
            # Bu kullanıcının değerlendireceği içerik sayısı
            n_ratings = max(5, np.random.randint(5, 30))
            n_ratings = min(n_ratings, len(items_df))
            
            # Değerlendirilecek içerikleri seç
            item_indices = np.random.choice(len(items_df), n_ratings, replace=False)
            
            for i in item_indices:
                # 1-5 arası puan, çoğunlukla olumlu puanlar (3+)
                if np.random.random() < 0.7:
                    # Olumlu puan (3-5)
                    rating = np.random.randint(3, 6)
                else:
                    # Olumsuz puan (1-2)
                    rating = np.random.randint(1, 3)
                
                ratings.append({
                    'user_id': user_id,
                    'item_id': items_df.iloc[i]['item_id'],
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        ratings_df = pd.DataFrame(ratings)
        
        logger.info(f"Örnek veri kümesi oluşturuldu: {len(items_df)} içerik, {len(ratings_df)} değerlendirme")
        return items_df, ratings_df
        
    def _fetch_turkish_content(self, count=50):
        """
        Özel olarak Türkçe içerik verisi çeker
        
        Args:
            count (int): Çekilecek içerik sayısı
            
        Returns:
            list: Türkçe film ve dizi verileri
        """
        turkish_content = []
        
        # Türkçe filmler
        try:
            response = requests.get(f"{self.base_url}/discover/movie", params={
                'api_key': self.api_key,
                'with_original_language': 'tr',
                'sort_by': 'popularity.desc',
                'page': 1
            })
            
            if response.status_code == 200:
                data = response.json()
                turkish_movies = data.get('results', [])
                
                for movie in turkish_movies[:count//2]:
                    # Verileri temizle
                    movie = self._clean_string_values(movie)
                    movie['content_type'] = 'movie'
                    turkish_content.append(movie)
        except Exception as e:
            logger.error(f"Türkçe film verisi çekme hatası: {str(e)}")
        
        # Türkçe diziler
        try:
            response = requests.get(f"{self.base_url}/discover/tv", params={
                'api_key': self.api_key,
                'with_original_language': 'tr',
                'sort_by': 'popularity.desc',
                'page': 1
            })
            
            if response.status_code == 200:
                data = response.json()
                turkish_tv = data.get('results', [])
                
                for tv in turkish_tv[:count//2]:
                    # Verileri temizle
                    tv = self._clean_string_values(tv)
                    tv['content_type'] = 'tv'
                    turkish_content.append(tv)
        except Exception as e:
            logger.error(f"Türkçe dizi verisi çekme hatası: {str(e)}")
        
        return turkish_content

    def _fetch_popular_movies(self, n_movies, min_vote_count):
        """
        Popüler filmleri çeker
        
        Args:
            n_movies (int): Çekilecek film sayısı
            min_vote_count (int): Minimum oy sayısı
            
        Returns:
            list: Popüler filmler
        """
        popular_movies = []
        page = 1
        while len(popular_movies) < n_movies:
            results = self.get_popular_movies(page=page)
            if 'results' in results:
                for movie in results['results']:
                    if movie['vote_count'] >= min_vote_count:
                        popular_movies.append(movie)
                    if len(popular_movies) >= n_movies:
                        break
                page += 1
            else:
                break
        return popular_movies[:n_movies]

    def _fetch_popular_tv_series(self, n_tv, min_vote_count):
        """
        Popüler dizileri çeker
        
        Args:
            n_tv (int): Çekilecek dizi sayısı
            min_vote_count (int): Minimum oy sayısı
            
        Returns:
            list: Popüler diziler
        """
        popular_tv = []
        page = 1
        while len(popular_tv) < n_tv:
            results = self.get_popular_tv_shows(page=page)
            if 'results' in results:
                for tv in results['results']:
                    if tv['vote_count'] >= min_vote_count:
                        popular_tv.append(tv)
                    if len(popular_tv) >= n_tv:
                        break
                page += 1
            else:
                break
        return popular_tv[:n_tv]

    def _fetch_movie_details(self, movie_id):
        """
        Film detaylarını çeker
        
        Args:
            movie_id (int): TMDB film ID
            
        Returns:
            dict: Film detayları
        """
        return self.get_movie_details(movie_id)

    def _fetch_tv_details(self, tv_id):
        """
        Dizi detaylarını çeker
        
        Args:
            tv_id (int): TMDB dizi ID
            
        Returns:
            dict: Dizi detayları
        """
        return self.get_tv_details(tv_id)

    def _generate_sample_ratings(self, items_df, n_users, min_ratings, max_ratings):
        """
        Örnek değerlendirmeler oluşturur
        
        Args:
            items_df (pd.DataFrame): İçerik DataFrame'i
            n_users (int): Kullanıcı sayısı
            min_ratings (int): Minimum değerlendirme sayısı
            max_ratings (int): Maksimum değerlendirme sayısı
            
        Returns:
            pd.DataFrame: Yapay değerlendirmeler
        """
        ratings = []
        
        for user_id in range(1, n_users + 1):
            # Her kullanıcı için rastgele sayıda içerik seç
            n_ratings = np.random.randint(min_ratings, max_ratings + 1)
            item_indices = np.random.choice(len(items_df), size=n_ratings, replace=False)
            
            for idx in item_indices:
                item_id = items_df.iloc[idx]['item_id']
                # 1-5 arası rastgele puan
                rating = np.random.randint(1, 6)
                # Rastgele tarih
                timestamp = int(datetime.now().timestamp())
                
                ratings.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        return pd.DataFrame(ratings) 