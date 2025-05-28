import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors

class KNNModel:
    def __init__(self, n_neighbors=5, window_size=100, threshold=1.5):
        """
        K-Nearest Neighbors modeli
        
        Args:
            n_neighbors: Komşu sayısı
            window_size: Pencere boyutu
            threshold: Eşik değeri
        """
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.threshold = threshold
        self.model = None
        self.sequences = None
        self.next_values = None
        
    def fit(self, values):
        """
        Modeli eğitir
        
        Args:
            values: Değerler dizisi
        """
        if len(values) <= self.window_size:
            return
            
        # Diziler oluştur
        sequences = []
        next_values = []
        
        for i in range(len(values) - self.window_size):
            seq = values[i:i+self.window_size]
            next_val = values[i+self.window_size]
            
            sequences.append(seq)
            next_values.append(next_val)
            
        if not sequences:
            return
            
        # Diziyi numpy dizisine dönüştür
        self.sequences = np.array(sequences)
        self.next_values = np.array(next_values)
        
        # KNN modelini oluştur ve eğit
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto')
        self.model.fit(self.sequences)
        
    def predict_next_value(self, sequence):
        """
        Bir sonraki değeri tahmin eder
        
        Args:
            sequence: Değerler dizisi
            
        Returns:
            tuple: (tahmini değer, eşik üstü olasılığı)
        """
        if self.model is None or len(sequence) < self.window_size:
            return None, 0.5
            
        # Son window_size değeri al
        query = np.array(sequence[-self.window_size:]).reshape(1, -1)
        
        # En yakın komşuları bul
        distances, indices = self.model.kneighbors(query)
        
        if len(indices[0]) == 0:
            return None, 0.5
            
        # En yakın komşuların sonraki değerleri
        neighbor_next_values = self.next_values[indices[0]]
        
        # Sonraki değeri tahmin et (ağırlıklı ortalama)
        weights = 1.0 / (distances[0] + 1e-5)  # Sıfıra bölme hatasını önle
        prediction = np.sum(neighbor_next_values * weights) / np.sum(weights)
        
        # Eşik üstü olasılığı
        above_count = sum(1 for v in neighbor_next_values if v >= self.threshold)
        above_prob = above_count / len(neighbor_next_values)
        
        return prediction, above_prob
