import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from .metrics import calculate_threshold_metrics, calculate_regression_metrics
import warnings

class BackTester:
    def __init__(self, predictor, data, test_size=0.2, threshold=1.5):
        """
        Geri Test sistemi
        
        Args:
            predictor: Tahmin modeli
            data: Test verisi
            test_size: Test seti oranı
            threshold: Eşik değeri
        """
        self.predictor = predictor
        self.data = data
        self.test_size = test_size
        self.threshold = threshold
        
        # Sonuçlar
        self.predictions = []
        self.actuals = []
        self.confidences = []
        self.metrics = None
        
    def run(self, window_size=100, step_size=1, show_progress=True):
        """
        Geri test yapar
        
        Args:
            window_size: Pencere boyutu
            step_size: Adım boyutu
            show_progress: İlerleme göster
            
        Returns:
            dict: Metrikler
        """
        # Test veri setini belirle
        n = len(self.data)
        train_size = int((1 - self.test_size) * n)
        train_data = self.data[:train_size]
        test_data = self.data[train_size:]
        
        if len(test_data) == 0:
            print("Test veri seti boş!")
            return None
            
        # Modeli eğit
        if hasattr(self.predictor, 'fit'):
            self.predictor.fit(train_data)
        
        # Tahminleri hesapla
        self.predictions = []
        self.actuals = []
        self.confidences = []
        
        # İlerleme çubuğu
        iterator = range(0, len(test_data) - 1, step_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Backtest")
            
        for i in iterator:
            if i < window_size:
                continue
                
            # Tahmin penceresi
            history = test_data[max(0, i-window_size):i]
            
            # Gerçek sonraki değer
            actual = test_data[i]
            
            try:
                # Tahmin yap
                result = self.predictor.predict_next_value(history)
                
                if isinstance(result, tuple) and len(result) >= 2:
                    pred_value, above_prob = result[0], result[1]
                    confidence = result[2] if len(result) >= 3 else 0.5
                else:
                    pred_value = result
                    above_prob = 1.0 if pred_value >= self.threshold else 0.0
                    confidence = 0.5
                
                # Sonuçları kaydet
                self.predictions.append(above_prob)
                self.actuals.append(actual)
                self.confidences.append(confidence)
            except Exception as e:
                print(f"Tahmin hatası (i={i}): {e}")
        
        # Metrikleri hesapla
        if self.predictions:
            self.metrics = calculate_threshold_metrics(self.actuals, 
                                                     [1.0 if p >= 0.5 else 0.0 for p in self.predictions],
                                                     threshold=self.threshold)
        else:
            print("Hiç geçerli tahmin yok!")
            self.metrics = None
            
        return self.metrics
    
    def plot_results(self, figsize=(12, 8)):
        """
        Sonuçları görselleştirir
        
        Args:
            figsize: Grafik boyutu
            
        Returns:
            matplotlib.figure.Figure: Grafik
        """
        if not self.predictions or not self.actuals:
            print("Gösterilecek sonuç yok!")
            return None
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            fig, axes = plt.subplots(2, 1, figsize=figsize)
            
            # Üst grafik: Tahmin vs Gerçek
            ax1 = axes[0]
            
            # Tahmin olasılıkları
            ax1.plot(self.predictions, 'bo-', alpha=0.5, label='Prediction (Prob)')
            
            # Gerçek değerler (ikili)
            actuals_binary = [1.0 if a >= self.threshold else 0.0 for a in self.actuals]
            ax1.plot(actuals_binary, 'ro-', alpha=0.5, label='Actual (Binary)')
            
            # Eşik çizgisi
            ax1.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Threshold (0.5)')
            
            # Güven skorları
            if self.confidences:
                ax1.plot(self.confidences, 'ko-', alpha=0.3, label='Confidence')
            
            ax1.set_xlabel('Test Sample')
            ax1.set_ylabel('Probability / Binary')
            ax1.set_title('Backtest Results')
            ax1.legend()
            
            # Alt grafik: Metrikleri göster
            ax2 = axes[1]
            
            if self.metrics:
                metrics = [
                    ('Accuracy', self.metrics['accuracy']),
                    ('Balanced Acc', self.metrics['balanced_accuracy']),
                    ('Precision', self.metrics['precision']),
                    ('Recall', self.metrics['recall']),
                    ('F1', self.metrics['f1'])
                ]
                
                bars = ax2.bar(range(len(metrics)), [m[1] for m in metrics], alpha=0.7)
                
                # Bar etiketleri
                ax2.set_xticks(range(len(metrics)))
                ax2.set_xticklabels([m[0] for m in metrics])
                
                # Değer etiketleri
                for i, bar in enumerate(bars):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"{metrics[i][1]:.3f}", ha='center', va='bottom', fontsize=10)
                
                # Şans çizgisi
                ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
                
                ax2.set_ylim(0, 1.1)
                ax2.set_title('Performance Metrics')
            
            plt.tight_layout()
            return fig
