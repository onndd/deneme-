import numpy as np
from collections import defaultdict

class PatternMatcher:
    def __init__(self, tolerance=0.08, threshold=1.5, pattern_lengths=[25, 35, 100, 500],
                 min_similarity_threshold=0.8, max_similar_patterns=10): # YENİ PARAMETRELER EKLENDİ
        """
        Toleranslı Örüntü Eşleştirme modeli
        
        Args:
            tolerance: Eşleşme toleransı (şu an _calculate_similarity içinde local_tolerance olarak sabit)
            threshold: Eşik değeri
            pattern_lengths: Kullanılacak örüntü uzunlukları
            min_similarity_threshold: Bir örüntünün benzer sayılması için gereken minimum benzerlik skoru
            max_similar_patterns: Bulunacak maksimum benzer örüntü sayısı
        """
        self.tolerance = tolerance # Bu parametre şu an _calculate_similarity içinde doğrudan kullanılmıyor.
        self.threshold = threshold
        self.pattern_lengths = pattern_lengths
        self.all_data = []
        self.min_similarity_threshold = min_similarity_threshold # YENİ ÖZNİTELİK
        self.max_similar_patterns = max_similar_patterns     # YENİ ÖZNİTELİK
    
    def fit(self, values):
        """
        Modeli veriye göre eğitir (tüm veriyi kaydeder)
        
        Args:
            values: Değerler dizisi
        """
        self.all_data = list(values)
    
    def _calculate_similarity(self, pattern1, pattern2):
        """
        İki örüntü arasındaki benzerliği hesaplar
        
        Args:
            pattern1: Birinci örüntü
            pattern2: İkinci örüntü
            
        Returns:
            float: Benzerlik skoru (0-1 arası)
        """
        if len(pattern1) != len(pattern2):
            return 0.0
        
        n = len(pattern1)
        diff_sum = 0.0
        
        for i in range(n):
            p1 = pattern1[i]
            p2 = pattern2[i]
            
            # Eşik değeri etrafında farklı tolerans
            if p1 < self.threshold:
                local_tolerance = 0.09  # Bu değerler sabit, __init__ içindeki self.tolerance'ı kullanabiliriz.
            else:
                local_tolerance = 0.07  # Bu değerler sabit, __init__ içindeki self.tolerance'ı kullanabiliriz.
                
            # Yüzde fark
            percent_diff = abs(p1 - p2) / max(0.01, p1)
            
            # Eğer fark tolerans içinde değilse
            if percent_diff > local_tolerance:
                diff_sum += percent_diff
        
        # Normalize edilmiş benzerlik (1 = tam benzer, 0 = hiç benzer değil)
        similarity = 1.0 - min(1.0, diff_sum / n)
        return similarity
    
    def _find_similar_patterns(self, current_pattern): # min_similarity ve max_count parametreleri kaldırıldı
        """
        Benzer örüntüleri bulur. 
        self.min_similarity_threshold ve self.max_similar_patterns özniteliklerini kullanır.
        
        Args:
            current_pattern: Mevcut örüntü
            
        Returns:
            list: Benzer örüntülerin sonraki değerlerinin listesi
        """
        n = len(current_pattern)
        similar_nexts = []
        similarities = []
        
        # Tüm veriyi tara
        for i in range(len(self.all_data) - n):
            candidate = self.all_data[i:i+n]
            
            # Benzerliği hesapla
            similarity = self._calculate_similarity(current_pattern, candidate)
            
            # Yeterince benzer mi? (Sınıf özniteliğini kullan)
            if similarity >= self.min_similarity_threshold:
                # Sonraki değeri kaydet
                if i+n < len(self.all_data):
                    next_value = self.all_data[i+n]
                    similar_nexts.append(next_value)
                    similarities.append(similarity)
                    
                    # Yeterli sayıda sonuç bulundu mu? (Sınıf özniteliğini kullan)
                    if len(similar_nexts) >= self.max_similar_patterns:
                        break
        
        return similar_nexts, similarities
    
    def predict_next_value(self, sequence): # min_similarity parametresi kaldırıldı
        """
        Bir sonraki değeri tahmin eder.
        self.min_similarity_threshold'u _find_similar_patterns içinde kullanır.
        
        Args:
            sequence: Değerler dizisi
            
        Returns:
            tuple: (tahmini değer, eşik üstü olasılığı, güven skoru)
        """
        results = []
        
        # Her bir örüntü uzunluğu için
        for length in self.pattern_lengths:
            if len(sequence) < length:
                continue
                
            # Güncel örüntüyü al
            current_pattern = sequence[-length:]
            
            # Benzer örüntüleri bul (artık ek parametreye ihtiyaç yok)
            similar_nexts, similarities = self._find_similar_patterns(current_pattern)
            
            if similar_nexts:
                # Benzerlik ile ağırlıklı ortalama
                weighted_sum = sum(v * s for v, s in zip(similar_nexts, similarities))
                total_similarity = sum(similarities)
                prediction = weighted_sum / total_similarity if total_similarity > 0 else None
                
                # Eşik üstü olasılığı
                above_count = sum(1 for v in similar_nexts if v >= self.threshold)
                above_prob = above_count / len(similar_nexts) if similar_nexts else 0.5
                
                # Güven skoru (benzerlik ortalaması)
                confidence = np.mean(similarities) if similarities else 0.0
                
                results.append((prediction, above_prob, confidence, len(similar_nexts)))
        
        if not results:
            return None, 0.5, 0.0
        
        # En güvenilir sonucu seç (en yüksek güven skoru ve en fazla örnek)
        best_result = max(results, key=lambda x: (x[2], x[3]))
        return best_result[0], best_result[1], best_result[2]