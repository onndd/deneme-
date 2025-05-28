import numpy as np
from collections import Counter
from data_processing.transformer import transform_to_categories

def extract_pattern_sequences(categories, pattern_lengths=[2, 3, 4, 5]):
    """
    Kategorilerden örüntü dizileri çıkarır
    
    Args:
        categories: Kategori kodlarının listesi
        pattern_lengths: Çıkarılacak örüntü uzunlukları
        
    Returns:
        dict: Her uzunluk için örüntü dizisi sayımları
    """
    patterns = {}
    
    for length in pattern_lengths:
        pattern_list = []
        for i in range(len(categories) - length + 1):
            pattern = tuple(categories[i:i+length])
            pattern_list.append(pattern)
        
        # Her bir örüntünün sayısını hesapla
        pattern_counts = Counter(pattern_list)
        patterns[length] = pattern_counts
    
    return patterns

def calculate_pattern_probabilities(patterns, smoothing=0.1):
    """
    Örüntü olasılıklarını hesaplar
    
    Args:
        patterns: Örüntü sayım sözlüğü
        smoothing: Düzleştirme parametresi
        
    Returns:
        dict: Her uzunluk için örüntü olasılıkları
    """
    probabilities = {}
    
    for length, pattern_counts in patterns.items():
        total = sum(pattern_counts.values())
        probs = {}
        
        for pattern, count in pattern_counts.items():
            # Laplace düzeltmesi ile olasılık hesapla
            prob = (count + smoothing) / (total + smoothing * len(pattern_counts))
            probs[pattern] = prob
        
        probabilities[length] = probs
    
    return probabilities

def encode_pattern_features(categories, window_sizes=[5, 10, 20, 50], pattern_lengths=[2, 3, 4]):
    """
    Örüntü özellikleri oluşturur
    
    Args:
        categories: Kategori kodlarının listesi
        window_sizes: Pencere boyutları
        pattern_lengths: Örüntü uzunlukları
        
    Returns:
        numpy.ndarray: Örüntü özellikler matrisi
    """
    n_samples = len(categories)
    n_features = len(window_sizes) * len(pattern_lengths) * 2
    features = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        feature_idx = 0
        
        for window in window_sizes:
            # Pencereyi al
            start = max(0, i - window)
            window_cats = categories[start:i]
            
            if len(window_cats) < max(pattern_lengths):
                feature_idx += len(pattern_lengths) * 2
                continue
            
            # Örüntüleri çıkar
            patterns = extract_pattern_sequences(window_cats, pattern_lengths)
            probabilities = calculate_pattern_probabilities(patterns)
            
            # Her uzunluk için özellikler
            for length in pattern_lengths:
                if length in patterns and length-1 < len(window_cats):
                    # Son örüntü
                    last_pattern = tuple(window_cats[-(length-1):])
                    
                    # Olası bir sonraki kategori için
                    max_prob = 0
                    entropy = 0
                    
                    for cat in set(window_cats):
                        next_pattern = last_pattern + (cat,)
                        prob = probabilities[length].get(next_pattern, 0)
                        max_prob = max(max_prob, prob)
                        if prob > 0:
                            entropy -= prob * np.log(prob)
                    
                    # Özellikleri kaydet
                    features[i, feature_idx] = max_prob
                    features[i, feature_idx + 1] = entropy
                
                feature_idx += 2
    
    return features

def find_similar_patterns(values, current_pattern, n_similar=5, tolerance=0.05):
    """
    Benzer örüntüleri bulur
    
    Args:
        values: Tüm değerlerin listesi
        current_pattern: Aranacak güncel örüntü
        n_similar: Döndürülecek maksimum benzer örüntü sayısı
        tolerance: Eşleşme toleransı
        
    Returns:
        list: Benzer örüntülerin sonraki değerlerinin listesi
    """
    pattern_length = len(current_pattern)
    similar_nexts = []
    
    # Tüm olası örüntüleri kontrol et
    for i in range(len(values) - pattern_length):
        candidate = values[i:i+pattern_length]
        
        # Toleranslı karşılaştırma
        is_similar = True
        for j in range(pattern_length):
            diff_ratio = abs(candidate[j] - current_pattern[j]) / max(0.01, current_pattern[j])
            
            # 1.5 eşiği etrafında özel tolerans
            if current_pattern[j] < 1.5:
                max_tolerance = 0.03  # %3 tolerans
            else:
                max_tolerance = 0.07  # %7 tolerans
                
            if diff_ratio > max_tolerance:
                is_similar = False
                break
        
        # Benzer örüntü bulundu, sonraki değeri kaydet
        if is_similar and i + pattern_length < len(values):
            similar_nexts.append(values[i + pattern_length])
            
            # Yeterli sayıda benzer örüntü bulundu mu?
            if len(similar_nexts) >= n_similar:
                break
    
    return similar_nexts

def extract_similarity_features(values, window_sizes=[5, 10, 15, 20]):
    """
    Benzerlik tabanlı özellikler çıkarır
    
    Args:
        values: Sayısal değerler listesi
        window_sizes: Kullanılacak pencere boyutları
        
    Returns:
        numpy.ndarray: Benzerlik özellikleri matrisi
    """
    n_samples = len(values)
    # Her pencere için: [sonraki_ortalama, üst_olasılık, min, max, std]
    n_features = len(window_sizes) * 5
    features = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        feature_idx = 0
        
        for window in window_sizes:
            if i < window:
                feature_idx += 5
                continue
            
            # Güncel örüntüyü al
            current_pattern = values[i-window:i]
            
            # Benzer örüntülerin sonraki değerlerini bul
            similar_nexts = find_similar_patterns(values[:i], current_pattern)
            
            if similar_nexts:
                # İstatistikler
                next_mean = np.mean(similar_nexts)
                above_prob = sum(1 for x in similar_nexts if x >= 1.5) / len(similar_nexts)
                next_min = np.min(similar_nexts)
                next_max = np.max(similar_nexts)
                next_std = np.std(similar_nexts)
                
                # Özellikleri kaydet
                features[i, feature_idx] = next_mean
                features[i, feature_idx + 1] = above_prob
                features[i, feature_idx + 2] = next_min
                features[i, feature_idx + 3] = next_max
                features[i, feature_idx + 4] = next_std
            
            feature_idx += 5
    
    return features
