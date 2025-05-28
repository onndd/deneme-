import numpy as np
from sklearn.neighbors import KernelDensity
import warnings

class DensityModel:
    def __init__(self, bandwidth=0.1, threshold=1.5, kernel='gaussian'):
        """
        Yoğunluk Analizi modeli
        
        Args:
            bandwidth: Bant genişliği
            threshold: Eşik değeri
            kernel: Kernel tipi
        """
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.kernel = kernel
        self.kde = None
        self.values = None
        
        # İstatistikler
        self.below_threshold_values = None
        self.above_threshold_values = None
    
    def fit(self, values):
        """
        Modeli veriye göre eğitir
        
        Args:
            values: Değerler dizisi
        """
        self.values = np.array(values).reshape(-1, 1)
        
        # Olası uyarıları bastır
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            self.kde.fit(self.values)
        
        # Eşik altı/üstü değerleri ayır
        self.below_threshold_values = [v for v in values if v < self.threshold]
        self.above_threshold_values = [v for v in values if v >= self.threshold]
    
    def _score_values(self, values):
        """
        Değerlerin skor matrisini hesaplar
        
        Args:
            values: Değerler dizisi
            
        Returns:
            numpy.ndarray: Skorlar
        """
        values = np.array(values).reshape(-1, 1)
        return np.exp(self.kde.score_samples(values))
    
    def predict_next_value(self, sequence=None, n_samples=100):
        """
        Bir sonraki değeri tahmin eder
        
        Args:
            sequence: Değerler dizisi (kullanılmaz)
            n_samples: Örnekleme sayısı
            
        Returns:
            tuple: (tahmini değer, eşik üstü olasılığı)
        """
        if self.kde is None:
            return None, 0.5
        
        # Örnekleme yap
        samples = self.kde.sample(n_samples=n_samples)
        samples = samples.flatten()
        
        # En yüksek yoğunluklu değer
        test_values = np.linspace(1.0, 10.0, 100).reshape(-1, 1)
        densities = self._score_values(test_values)
        prediction = test_values[np.argmax(densities)][0]
        
        # Eşik üstü olasılığı
        if not self.above_threshold_values or not self.below_threshold_values:
            above_prob = 0.5
        else:
            above_count = len(self.above_threshold_values)
            below_count = len(self.below_threshold_values)
            total = above_count + below_count
            above_prob = above_count / total if total > 0 else 0.5
        
        return prediction, above_prob
    
    def get_high_density_regions(self, min_samples=5, threshold_factor=0.5):
        """
        Yüksek yoğunluklu bölgeleri bulur
        
        Args:
            min_samples: Minimum örnek sayısı
            threshold_factor: Yoğunluk eşik faktörü
            
        Returns:
            list: Yüksek yoğunluklu bölgeler (min, max)
        """
        if self.kde is None or self.values is None:
            return []
        
        # Veri aralığında değerleri üret
        min_val, max_val = np.min(self.values), np.max(self.values)
        x = np.linspace(min_val, max_val, 1000).reshape(-1, 1)
        
        # Yoğunlukları hesapla
        densities = self._score_values(x)
        
        # Yoğunluk eşiği (maksimum yoğunluğun bir oranı)
        density_threshold = np.max(densities) * threshold_factor
        
        # Yüksek yoğunluklu bölgeleri bul
        high_density_regions = []
        in_region = False
        region_start = None
        
        for i, (val, density) in enumerate(zip(x, densities)):
            if density >= density_threshold:
                if not in_region:
                    in_region = True
                    region_start = val[0]
            else:
                if in_region:
                    in_region = False
                    region_end = x[i-1][0]
                    
                    # Bölgede yeterli örnek var mı?
                    region_samples = sum(1 for v in self.values if region_start <= v <= region_end)
                    if region_samples >= min_samples:
                        high_density_regions.append((region_start, region_end))
        
        # Son bölge
        if in_region:
            region_end = x[-1][0]
            region_samples = sum(1 for v in self.values if region_start <= v <= region_end)
            if region_samples >= min_samples:
                high_density_regions.append((region_start, region_end))
        
        return high_density_regions
