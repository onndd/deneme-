import numpy as np
import pandas as pd
from data_processing.transformer import (
    get_value_category, get_step_category,
    transform_to_categories, transform_to_step_categories,
    fuzzy_membership, VALUE_CATEGORIES
)

def encode_one_hot(categories, all_categories=None):
    """
    Kategori listesini one-hot kodlamaya dönüştürür
    
    Args:
        categories: Kategori kodlarının listesi
        all_categories: Tüm olası kategorilerin listesi (None=otomatik)
        
    Returns:
        numpy.ndarray: One-hot kodlama matrisi
    """
    if all_categories is None:
        all_categories = list(VALUE_CATEGORIES.keys())
    
    # Benzersiz kategorileri bul
    unique_categories = sorted(all_categories)
    
    # One-hot kodlama oluştur
    n_samples = len(categories)
    n_features = len(unique_categories)
    encoded = np.zeros((n_samples, n_features))
    
    for i, category in enumerate(categories):
        try:
            j = unique_categories.index(category)
            encoded[i, j] = 1
        except ValueError:
            # Kategori listede yoksa, en yakın kategoriyi bul
            pass
    
    return encoded

def encode_fuzzy_membership(values, categories=None):
    """
    Değerleri bulanık üyelik derecelerine göre kodlar
    
    Args:
        values: Sayısal değerler listesi
        categories: Kategori kodları (None=tüm kategoriler)
        
    Returns:
        numpy.ndarray: Bulanık üyelik dereceleri matrisi
    """
    if categories is None:
        categories = list(VALUE_CATEGORIES.keys())
    
    n_samples = len(values)
    n_features = len(categories)
    fuzzy_encoded = np.zeros((n_samples, n_features))
    
    for i, value in enumerate(values):
        for j, category in enumerate(categories):
            fuzzy_encoded[i, j] = fuzzy_membership(value, category)
    
    return fuzzy_encoded

def encode_category_counts(categories, window_sizes=[5, 10, 20, 50]):
    """
    Çeşitli pencere boyutlarında kategori sayılarını kodlar
    
    Args:
        categories: Kategori kodlarının listesi
        window_sizes: Pencere boyutları listesi
        
    Returns:
        numpy.ndarray: Her kategori için sayım matrisi
    """
    unique_categories = sorted(set(categories))
    n_samples = len(categories)
    n_windows = len(window_sizes)
    n_categories = len(unique_categories)
    
    counts = np.zeros((n_samples, n_windows * n_categories))
    
    for i in range(n_samples):
        for j, window in enumerate(window_sizes):
            # Bu konumdan önceki pencereyi al
            start = max(0, i - window)
            window_cats = categories[start:i]
            
            # Her kategori için sayım
            for k, category in enumerate(unique_categories):
                cat_count = window_cats.count(category)
                counts[i, j * n_categories + k] = cat_count / max(1, len(window_cats))
    
    return counts

def encode_threshold_statistics(values, threshold=1.5, window_sizes=[5, 10, 20, 50, 100]):
    """
    Eşik değerine göre istatistiksel özellikler oluşturur
    
    Args:
        values: Sayısal değerler listesi
        threshold: Eşik değeri
        window_sizes: Pencere boyutları listesi
        
    Returns:
        numpy.ndarray: İstatistiksel özellikler matrisi
    """
    n_samples = len(values)
    n_windows = len(window_sizes)
    # Her pencere için: [üstünde_oran, altında_oran, üst_ortalama, alt_ortalama]
    n_features = n_windows * 4
    
    stats = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        for j, window in enumerate(window_sizes):
            # Bu konumdan önceki pencereyi al
            start = max(0, i - window)
            window_vals = values[start:i]
            
            if not window_vals:
                continue
                
            # Eşik üstü/altı değerler
            above = [v for v in window_vals if v >= threshold]
            below = [v for v in window_vals if v < threshold]
            
            # Özellikler
            above_ratio = len(above) / len(window_vals) if window_vals else 0
            below_ratio = len(below) / len(window_vals) if window_vals else 0
            
            above_mean = np.mean(above) if above else 0
            below_mean = np.mean(below) if below else 0
            
            # Değerleri kaydet
            feature_idx = j * 4
            stats[i, feature_idx] = above_ratio
            stats[i, feature_idx + 1] = below_ratio
            stats[i, feature_idx + 2] = above_mean
            stats[i, feature_idx + 3] = below_mean
    
    return stats

def extract_categorical_features(values, window_sizes=[5, 10, 20, 50, 100]):
    """
    Tüm kategorik özellikleri çıkarır
    
    Args:
        values: Sayısal değerler listesi
        window_sizes: Pencere boyutları listesi
        
    Returns:
        numpy.ndarray: Özellikler matrisi
    """
    # Kategorilere dönüştür
    categories = transform_to_categories(values)
    step_categories = transform_to_step_categories(values)
    
    # Özellikler
    one_hot = encode_one_hot(categories)
    fuzzy = encode_fuzzy_membership(values)
    cat_counts = encode_category_counts(categories, window_sizes)
    threshold_stats = encode_threshold_statistics(values, 1.5, window_sizes)
    
    # Özellikleri birleştir
    return np.hstack([one_hot, fuzzy, cat_counts, threshold_stats])
