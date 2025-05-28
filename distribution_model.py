import numpy as np
from scipy import stats
import warnings

class DistributionModel:
    def __init__(self, window_size=300, threshold=1.5):
        """
        Dağılım Analizi modeli
        
        Args:
            window_size: Pencere boyutu
            threshold: Eşik değeri
        """
        self.window_size = window_size
        self.threshold = threshold
        
        # İstatistikler
        self.global_stats = None
        self.above_threshold_ratio = 0.5
        self.distribution_params = {}
        
    def fit(self, values):
        """
        Modeli eğitir
        
        Args:
            values: Değerler dizisi
        """
        if not values:
            return
            
        # Temel istatistikler
        self.global_stats = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'skewness': stats.skew(values) if len(values) > 2 else 0,
            'kurtosis': stats.kurtosis(values) if len(values) > 2 else 0
        }
        
        # Eşik üstü oranı
        above_count = sum(1 for v in values if v >= self.threshold)
        self.above_threshold_ratio = above_count / len(values)
        
        # Dağılım uydurma
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Normal dağılım
            try:
                norm_params = stats.norm.fit(values)
                self.distribution_params['norm'] = norm_params
            except:
                pass
                
            # Lognormal dağılım
            try:
                lognorm_params = stats.lognorm.fit(values)
                self.distribution_params['lognorm'] = lognorm_params
            except:
                pass
                
            # Gamma dağılım
            try:
                gamma_params = stats.gamma.fit(values)
                self.distribution_params['gamma'] = gamma_params
            except:
                pass
    
    def predict_next_value(self, sequence=None):
        """
        Bir sonraki değeri tahmin eder
        
        Args:
            sequence: Değerler dizisi (kullanılmıyor)
            
        Returns:
            tuple: (tahmini değer, eşik üstü olasılığı)
        """
        if self.global_stats is None:
            return None, 0.5
            
        # En iyi uyan dağılım için beklenen değer
        if 'lognorm' in self.distribution_params:
            # Lognormal dağılım için beklenen değer
            shape, loc, scale = self.distribution_params['lognorm']
            prediction = np.exp(loc + 0.5 * shape**2) * scale
        elif 'gamma' in self.distribution_params:
            # Gamma dağılım için beklenen değer
            shape, loc, scale = self.distribution_params['gamma']
            prediction = shape * scale + loc
        elif 'norm' in self.distribution_params:
            # Normal dağılım için beklenen değer
            loc, scale = self.distribution_params['norm']
            prediction = loc
        else:
            # Dağılım uydurma başarısız olduğunda ortalama
            prediction = self.global_stats['mean']
        
        # Eşik üstü olasılığı
        return prediction, self.above_threshold_ratio
    
    def generate_samples(self, n_samples=1000, distribution='best'):
        """
        Dağılıma göre rastgele örnekler üretir
        
        Args:
            n_samples: Örneklem sayısı
            distribution: Kullanılacak dağılım ('norm', 'lognorm', 'gamma', 'best')
            
        Returns:
            numpy.ndarray: Örnekler
        """
        if not self.distribution_params:
            if self.global_stats:
                # İstatistiklere göre normal dağılım
                return np.random.normal(
                    self.global_stats['mean'],
                    self.global_stats['std'],
                    n_samples
                )
            return np.ones(n_samples) * 1.5
            
        # En iyi dağılımı seç
        if distribution == 'best':
            if 'lognorm' in self.distribution_params:
                distribution = 'lognorm'
            elif 'gamma' in self.distribution_params:
                distribution = 'gamma'
            else:
                distribution = 'norm'
                
        # Seçilen dağılımdan örnekle
        if distribution == 'lognorm' and 'lognorm' in self.distribution_params:
            return stats.lognorm.rvs(*self.distribution_params['lognorm'], size=n_samples)
        elif distribution == 'gamma' and 'gamma' in self.distribution_params:
            return stats.gamma.rvs(*self.distribution_params['gamma'], size=n_samples)
        elif distribution == 'norm' and 'norm' in self.distribution_params:
            return stats.norm.rvs(*self.distribution_params['norm'], size=n_samples)
        
        # Dağılım bulunamadı, normal dağılım varsay
        return np.random.normal(
            self.global_stats['mean'] if self.global_stats else 1.5,
            self.global_stats['std'] if self.global_stats else 0.5,
            n_samples
        )
