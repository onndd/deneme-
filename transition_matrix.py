import numpy as np
from collections import defaultdict, Counter
import pandas as pd

class TransitionMatrixModel:
    def __init__(self, window_size=50, threshold=1.5, use_categories=True):
        """
        Geçiş Matrisi modeli
        
        Args:
            window_size: Analiz penceresi boyutu
            threshold: Eşik değeri
            use_categories: Kategoriler kullanılsın mı?
        """
        self.window_size = window_size
        self.threshold = threshold
        self.use_categories = use_categories
        
        # Geçiş matrisleri
        self.first_order_matrix = defaultdict(Counter)  # A -> B
        self.second_order_matrix = defaultdict(Counter)  # (A,B) -> C
        
        # İstatistikler
        self.global_distribution = Counter()
        self.above_threshold_after = defaultdict(Counter)  # Son durum -> Eşik üstü mü?
        
    def _get_category(self, value):
        """
        Değerin kategorisini belirler
        
        Args:
            value: Sayısal değer
            
        Returns:
            str: Kategori kodu
        """
        if self.use_categories:
            from data_processing.transformer import get_value_category
            return get_value_category(value)
        else:
            # Kategori kullanılmıyorsa, değeri yuvarla
            return round(value, 2)
    
    def fit(self, values):
        """
        Modeli eğitir
        
        Args:
            values: Değerler dizisi
        """
        if len(values) < 3:
            return
            
        # Kategorilere dönüştür
        if self.use_categories:
            categories = [self._get_category(v) for v in values]
        else:
            categories = [round(v, 2) for v in values]
            
        # Global dağılım
        self.global_distribution = Counter(categories)
        
        # Birinci derece geçişler (A -> B)
        for i in range(len(categories) - 1):
            current = categories[i]
            next_cat = categories[i + 1]
            self.first_order_matrix[current][next_cat] += 1
            
            # Eşik üstü/altı istatistikleri
            is_above = values[i + 1] >= self.threshold
            self.above_threshold_after[current][is_above] += 1
            
        # İkinci derece geçişler (A,B -> C)
        for i in range(len(categories) - 2):
            current_pair = (categories[i], categories[i + 1])
            next_cat = categories[i + 2]
            self.second_order_matrix[current_pair][next_cat] += 1
    
    def predict_next_value(self, sequence):
        """
        Bir sonraki değeri tahmin eder
        
        Args:
            sequence: Değerler dizisi
            
        Returns:
            tuple: (tahmini değer, eşik üstü olasılığı)
        """
        if len(sequence) < 2:
            # Tahmin için yeterli veri yok
            return None, 0.5
            
        # Son değerlerin kategorilerini al
        if self.use_categories:
            last_category = self._get_category(sequence[-1])
            second_last_category = self._get_category(sequence[-2]) if len(sequence) >= 2 else None
        else:
            last_category = round(sequence[-1], 2)
            second_last_category = round(sequence[-2], 2) if len(sequence) >= 2 else None
            
        # İkinci derece geçiş olasılıkları
        if second_last_category is not None:
            pair = (second_last_category, last_category)
            if pair in self.second_order_matrix and sum(self.second_order_matrix[pair].values()) > 0:
                # İkinci derece tahmini
                next_counts = self.second_order_matrix[pair]
                total = sum(next_counts.values())
                
                # En olası kategori
                most_likely = max(next_counts.items(), key=lambda x: x[1])[0]
                
                # Eşik üstü olasılığı
                if self.use_categories:
                    from data_processing.transformer import VALUE_CATEGORIES
                    above_prob = sum(count for cat, count in next_counts.items()
                                    if VALUE_CATEGORIES.get(cat, (0, 0))[0] >= self.threshold) / total
                    
                    # Tahmini değer (kategori orta noktası)
                    min_val, max_val = VALUE_CATEGORIES.get(most_likely, (1.0, 1.5))
                    prediction = (min_val + max_val) / 2
                else:
                    # Sayısal değerler için
                    above_prob = sum(count for cat, count in next_counts.items()
                                   if cat >= self.threshold) / total
                    prediction = most_likely
                
                return prediction, above_prob
                
        # Birinci derece geçiş olasılıkları
        if last_category in self.first_order_matrix and sum(self.first_order_matrix[last_category].values()) > 0:
            # Birinci derece tahmini
            next_counts = self.first_order_matrix[last_category]
            total = sum(next_counts.values())
            
            # En olası kategori
            most_likely = max(next_counts.items(), key=lambda x: x[1])[0]
            
            # Eşik üstü olasılığı
            if self.use_categories:
                from data_processing.transformer import VALUE_CATEGORIES
                above_prob = sum(count for cat, count in next_counts.items()
                                if VALUE_CATEGORIES.get(cat, (0, 0))[0] >= self.threshold) / total
                
                # Tahmini değer (kategori orta noktası)
                min_val, max_val = VALUE_CATEGORIES.get(most_likely, (1.0, 1.5))
                prediction = (min_val + max_val) / 2
            else:
                # Sayısal değerler için
                above_prob = sum(count for cat, count in next_counts.items()
                               if cat >= self.threshold) / total
                prediction = most_likely
            
            return prediction, above_prob
            
        # Geçiş matrisi eksikse, global dağılımı kullan
        if sum(self.global_distribution.values()) > 0:
            # Global dağılıma göre tahmin
            most_likely = max(self.global_distribution.items(), key=lambda x: x[1])[0]
            total = sum(self.global_distribution.values())
            
            # Eşik üstü olasılığı
            if self.use_categories:
                from data_processing.transformer import VALUE_CATEGORIES
                above_prob = sum(count for cat, count in self.global_distribution.items()
                                if VALUE_CATEGORIES.get(cat, (0, 0))[0] >= self.threshold) / total
                
                # Tahmini değer (kategori orta noktası)
                min_val, max_val = VALUE_CATEGORIES.get(most_likely, (1.0, 1.5))
                prediction = (min_val + max_val) / 2
            else:
                # Sayısal değerler için
                above_prob = sum(count for cat, count in self.global_distribution.items()
                               if cat >= self.threshold) / total
                prediction = most_likely
            
            return prediction, above_prob
        
        # Hiçbir veri yoksa varsayılan değerleri döndür
        return 1.5, 0.5
        
    def get_transition_matrix(self, as_dataframe=True):
        """
        Geçiş matrisini döndürür
        
        Args:
            as_dataframe: DataFrame olarak döndürülsün mü?
            
        Returns:
            dict veya DataFrame: Geçiş matrisi
        """
        if not self.first_order_matrix:
            return {} if not as_dataframe else pd.DataFrame()
            
        # Tüm kategorileri bul
        all_categories = set()
        for src, dests in self.first_order_matrix.items():
            all_categories.add(src)
            all_categories.update(dests.keys())
            
        all_categories = sorted(all_categories)
        
        if not as_dataframe:
            # Sözlük olarak döndür
            matrix = {}
            for src in all_categories:
                matrix[src] = {}
                total = sum(self.first_order_matrix[src].values())
                
                for dest in all_categories:
                    if total > 0:
                        matrix[src][dest] = self.first_order_matrix[src][dest] / total
                    else:
                        matrix[src][dest] = 0.0
                        
            return matrix
        else:
            # DataFrame olarak döndür
            matrix = pd.DataFrame(0.0, index=all_categories, columns=all_categories)
            
            for src in all_categories:
                total = sum(self.first_order_matrix[src].values())
                
                if total > 0:
                    for dest, count in self.first_order_matrix[src].items():
                        matrix.loc[src, dest] = count / total
                        
            return matrix
