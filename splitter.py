import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def create_sequences(data, seq_length):
    """
    Zaman serisi verilerinden sıralı diziler oluşturur
    
    Args:
        data: Sayısal veri dizisi
        seq_length: Dizi uzunluğu
        
    Returns:
        tuple: X (input sequences) ve y (hedef değerler)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def time_series_split(data, n_splits=5, test_size=0.2):
    """
    Zaman serisi verilerini eğitim/test olarak böler
    
    Args:
        data: Veri dizisi
        n_splits: Kaç farklı bölünme yapılacağı
        test_size: Test setinin büyüklük oranı
        
    Returns:
        list: (train_indices, test_indices) çiftlerinin listesi
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(data) * test_size))
    splits = []
    
    for train_index, test_index in tscv.split(data):
        splits.append((train_index, test_index))
        
    return splits

def create_above_threshold_target(data, threshold=1.5):
    """
    Eşik değeri üstünde/altında ikili hedef değerler oluşturur
    
    Args:
        data: Veri dizisi
        threshold: Eşik değeri (varsayılan=1.5)
        
    Returns:
        numpy.ndarray: İkili hedef değerler (0=altında, 1=üstünde)
    """
    return (np.array(data) >= threshold).astype(int)

def create_category_target(data):
    """
    Kategori hedef değerleri oluşturur
    
    Args:
        data: Veri dizisi
        
    Returns:
        list: Kategori kodlarının listesi
    """
    from .transformer import transform_to_categories
    return transform_to_categories(data)

def split_recent(data, recent_size=1000):
    """
    Verileri son N kayıt ve önceki kayıtlar olarak böler
    
    Args:
        data: Veri dizisi
        recent_size: Son kaç kaydın "yeni" kabul edileceği
        
    Returns:
        tuple: (older_data, recent_data)
    """
    if len(data) <= recent_size:
        return np.array([]), np.array(data)
    
    older_data = np.array(data[:-recent_size])
    recent_data = np.array(data[-recent_size:])
    
    return older_data, recent_data
