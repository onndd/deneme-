import numpy as np
from scipy import stats

def calculate_basic_stats(values, window_sizes=[10, 20, 50, 100, 200]):
    """
    Temel istatistiksel özellikleri hesaplar
    
    Args:
        values: Sayısal değerler listesi
        window_sizes: Pencere boyutları
        
    Returns:
        numpy.ndarray: İstatistiksel özellikler matrisi
    """
    n_samples = len(values)
    # Her pencere için: [ortalama, medyan, std, min, max, çarpıklık, basıklık]
    n_features = len(window_sizes) * 7
    features = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        feature_idx = 0
        
        for window in window_sizes:
            # Pencereyi al
            start = max(0, i - window)
            window_vals = values[start:i]
            
            if not window_vals:
                feature_idx += 7
                continue
            
            # İstatistikler
            mean = np.mean(window_vals)
            median = np.median(window_vals)
            std = np.std(window_vals)
            min_val = np.min(window_vals)
            max_val = np.max(window_vals)
            
            # Çarpıklık ve basıklık
            skewness = stats.skew(window_vals) if len(window_vals) > 2 else 0
            kurtosis = stats.kurtosis(window_vals) if len(window_vals) > 2 else 0
            
            # Özellikleri kaydet
            features[i, feature_idx] = mean
            features[i, feature_idx + 1] = median
            features[i, feature_idx + 2] = std
            features[i, feature_idx + 3] = min_val
            features[i, feature_idx + 4] = max_val
            features[i, feature_idx + 5] = skewness
            features[i, feature_idx + 6] = kurtosis
            
            feature_idx += 7
    
    return features

def calculate_threshold_runs(values, threshold=1.5, max_run_length=10):
    """
    Eşik değeri üzerinde/altında ardışık değer sayılarını hesaplar
    
    Args:
        values: Sayısal değerler listesi
        threshold: Eşik değeri
        max_run_length: Maksimum dizi uzunluğu
        
    Returns:
        numpy.ndarray: Dizi uzunluğu özellikleri
    """
    n_samples = len(values)
    # [üstünde_ardışık, altında_ardışık, üst_run_max, alt_run_max]
    n_features = 4
    features = np.zeros((n_samples, n_features))
    
    # Eşik üstü/altı değerleri belirle
    above_threshold = [1 if x >= threshold else 0 for x in values]
    
    for i in range(n_samples):
        # Güncel durumda ardışık değer sayısı
        current_above_run = 0
        current_below_run = 0
        
        # Geriye doğru kontrol et
        for j in range(i-1, max(0, i-max_run_length)-1, -1):
            if above_threshold[j] == 1:
                current_above_run += 1
                if current_below_run > 0:
                    break
            else:
                current_below_run += 1
                if current_above_run > 0:
                    break
        
        # Maksimum ardışık değer sayıları
        max_above_run = 0
        max_below_run = 0
        current_run = 1
        
        for j in range(1, min(i+1, max_run_length)):
            if i-j < 0:
                break
                
            if above_threshold[i-j] == above_threshold[i-j+1]:
                current_run += 1
            else:
                if above_threshold[i-j+1] == 1:
                    max_above_run = max(max_above_run, current_run)
                else:
                    max_below_run = max(max_below_run, current_run)
                current_run = 1
        
        # Son diziyi kontrol et
        if i > 0:
            if above_threshold[0] == 1:
                max_above_run = max(max_above_run, current_run)
            else:
                max_below_run = max(max_below_run, current_run)
        
        # Özellikleri kaydet
        features[i, 0] = current_above_run
        features[i, 1] = current_below_run
        features[i, 2] = max_above_run
        features[i, 3] = max_below_run
    
    return features

def calculate_trend_features(values, window_sizes=[10, 20, 50, 100]):
    """
    Trend ve mevsimsellik özellikleri hesaplar
    
    Args:
        values: Sayısal değerler listesi
        window_sizes: Pencere boyutları
        
    Returns:
        numpy.ndarray: Trend özellikleri matrisi
    """
    n_samples = len(values)
    # Her pencere için: [eğim, otokorelasyon, trend_güç]
    n_features = len(window_sizes) * 3
    features = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        feature_idx = 0
        
        for window in window_sizes:
            # Pencereyi al
            start = max(0, i - window)
            window_vals = np.array(values[start:i])
            
            if len(window_vals) < 2:
                feature_idx += 3
                continue
            
            # Zaman indeksleri
            time_idx = np.arange(len(window_vals))
            
            # Eğim hesapla (doğrusal regresyon)
            if len(window_vals) > 1:
                slope, _, _, _, _ = stats.linregress(time_idx, window_vals)
            else:
                slope = 0
            
            # Otokorelasyon (lag=1)
            if len(window_vals) > 1:
                autocorr = np.corrcoef(window_vals[:-1], window_vals[1:])[0, 1] if len(window_vals) > 1 else 0
            else:
                autocorr = 0
            
            # Trend gücü (eğim / standart sapma)
            trend_strength = slope / np.std(window_vals) if np.std(window_vals) > 0 else 0
            
            # Özellikleri kaydet
            features[i, feature_idx] = slope
            features[i, feature_idx + 1] = autocorr
            features[i, feature_idx + 2] = trend_strength
            
            feature_idx += 3
    
    return features

def extract_statistical_features(values):
    """
    Tüm istatistiksel özellikleri çıkarır
    
    Args:
        values: Sayısal değerler listesi
        
    Returns:
        numpy.ndarray: İstatistiksel özellikler matrisi
    """
    # Temel istatistikler
    basic_stats = calculate_basic_stats(values)
    
    # Eşik değeri dizi özellikleri
    threshold_runs = calculate_threshold_runs(values)
    
    # Trend özellikleri
    trend_features = calculate_trend_features(values)
    
    # Özellikleri birleştir
    return np.hstack([basic_stats, threshold_runs, trend_features])
