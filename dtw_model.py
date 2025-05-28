import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

class DTWModel:
    def __init__(self, window_size=10, max_neighbors=5, threshold=1.5, max_distance=0.5):
        """
        Dynamic Time Warping modeli
        
        Args:
            window_size: Karşılaştırılacak dizi uzunluğu
            max_neighbors: Maksimum komşu sayısı
            threshold: Eşik değeri
            max_distance: Maksimum DTW mesafesi
        """
        self.window_size = window_size
        self.max_neighbors = max_neighbors
        self.threshold = threshold
        self.max_distance = max_distance
        self.sequences = []
        self.next_values = []
        
    def fit(self, values):
        """
        Modeli eğitir
        
        Args:
            values: Değerler dizisi
        """
        if len(values) <= self.window_size:
            return
            
        # Diziler oluştur
        self.sequences = []
        self.next_values = []
        
        for i in range(len(values) - self.window_size):
            seq = values[i:i+self.window_size]
            next_val = values[i+self.window_size]
            
            self.sequences.append(seq)
            self.next_values.append(next_val)
    
    def _get_nearest_sequences(self, query_sequence, top_n=None):
        """
        En yakın dizileri bulur
        
        Args:
            query_sequence: Sorgu dizisi
            top_n: Döndürülecek maksimum sonuç sayısı
            
        Returns:
            list: (mesafe, indeks, sonraki değer) üçlülerinin listesi
        """
        if not self.sequences:
            return []
            
        if top_n is None:
            top_n = self.max_neighbors
            
        # DTW mesafelerini hesapla
        distances = []
        
        for i, seq in enumerate(self.sequences):
            # FastDTW mesafesini hesapla
            distance, _ = fastdtw(query_sequence, seq, dist=euclidean)
            
            # Normalize et
            distance /= len(query_sequence)
            
            if distance <= self.max_distance:
                distances.append((distance, i, self.next_values[i]))
        
        # Mesafeye göre sırala
        distances.sort(key=lambda x: x[0])
        
        # En yakın top_n sonucu döndür
        return distances[:top_n]
    
    def predict_next_value(self, sequence):
        """
        Bir sonraki değeri tahmin eder
        
        Args:
            sequence: Değerler dizisi
            
        Returns:
            tuple: (tahmini değer, eşik üstü olasılığı)
        """
        if not self.sequences or len(sequence) < self.window_size:
            return None, 0.5
            
        # Son window_size değeri al
        query = sequence[-self.window_size:]
        
        # En yakın dizileri bul
        nearest = self._get_nearest_sequences(query)
        
        if not nearest:
            return None, 0.5
            
        # Ağırlıklı tahmin
        total_weight = 0
        weighted_sum = 0
        above_weight = 0
        
        for distance, _, next_value in nearest:
            # Uzaklık tersini ağırlık olarak kullan
            weight = 1.0 / (distance + 1e-5)
            total_weight += weight
            weighted_sum += next_value * weight
            
            # Eşik üstü ağırlık
            if next_value >= self.threshold:
                above_weight += weight
        
        if total_weight == 0:
            return None, 0.5
            
        # Ağırlıklı ortalama
        prediction = weighted_sum / total_weight
        
        # Eşik üstü olasılığı
        above_prob = above_weight / total_weight
        
        return prediction, above_prob
