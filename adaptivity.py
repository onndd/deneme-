import numpy as np
from collections import defaultdict, deque
import pandas as pd
import matplotlib.pyplot as plt
import warnings

class AdaptiveEnsemble:
    def __init__(self, models=None, window_size=100, threshold=1.5, adaptation_rate=0.1):
        """
        Adaptif Ensemble modeli
        
        Args:
            models: Alt modeller sözlüğü
            window_size: Pencere boyutu
            threshold: Eşik değeri
            adaptation_rate: Adaptasyon hızı (0-1 arası)
        """
        self.models = models or {}
        self.window_size = window_size
        self.threshold = threshold
        self.adaptation_rate = adaptation_rate
        
        # Model ağırlıkları
        self.weights = {name: 1.0 for name in self.models}
        
        # Performans geçmişi
        self.performance_history = {name: deque(maxlen=window_size) for name in self.models}
        
        # Global performans (tüm veri üzerinde)
        self.global_performance = {name: {'correct': 0, 'total': 0} for name in self.models}
        
        # Algoritma değişimi tespiti
        self.algorithm_change_detected = False
        self.recent_accuracy = deque(maxlen=50)
    
    def add_model(self, name, model, weight=1.0):
        """
        Yeni bir model ekler
        
        Args:
            name: Model adı
            model: Model nesnesi
            weight: Başlangıç ağırlığı
        """
        self.models[name] = model
        self.weights[name] = weight
        self.performance_history[name] = deque(maxlen=self.window_size)
        self.global_performance[name] = {'correct': 0, 'total': 0}
    
    def update_weights(self):
        """
        Model ağırlıklarını günceller
        """
        # Her modelin son performansını hesapla
        for name in self.models:
            history = self.performance_history[name]
            
            if not history:
                continue
                
            # Son penceredeki performans
            correct = sum(1 for x in history if x)
            total = len(history)
            
            accuracy = correct / total if total > 0 else 0.5
            
            # Ağırlıkları güncelle
            new_weight = accuracy
            
            # Adaptasyon hızı ile yumuşat
            self.weights[name] = (1.0 - self.adaptation_rate) * self.weights[name] + \
                                self.adaptation_rate * new_weight
        
        # Ağırlıkları normalize et
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for name in self.weights:
                self.weights[name] /= total_weight
    
    def record_performance(self, predictions, actual_value):
        """
        Model performanslarını kaydeder
        
        Args:
            predictions: Model tahminleri sözlüğü {model_adı: (değer, olasılık)}
            actual_value: Gerçek değer
        """
        # Gerçek eşik durumu
        actual_above = actual_value >= self.threshold
        
        # Her modelin performansını kaydet
        for name, (_, prob) in predictions.items():
            if name not in self.models:
                continue
                
            # Tahmini eşik durumu
            predicted_above = prob >= 0.5
            
            # Doğru mu?
            correct = (predicted_above == actual_above)
            
            # Performans geçmişine ekle
            self.performance_history[name].append(correct)
            
            # Global performansı güncelle
            self.global_performance[name]['total'] += 1
            if correct:
                self.global_performance[name]['correct'] += 1
                
        # Ensemble performansını kaydet
        self.recent_accuracy.append(1 if self._ensemble_correct(predictions, actual_value) else 0)
        
        # Algoritma değişimi tespiti
        if len(self.recent_accuracy) >= 20:
            recent_acc = sum(self.recent_accuracy) / len(self.recent_accuracy)
            if recent_acc < 0.45:  # Şans seviyesinin altı
                self.algorithm_change_detected = True
            else:
                self.algorithm_change_detected = False
                
        # Ağırlıkları güncelle
        self.update_weights()
    
    def _ensemble_correct(self, predictions, actual_value):
        """
        Ensemble'ın doğru tahmin yapıp yapmadığını kontrol eder
        
        Args:
            predictions: Model tahminleri sözlüğü
            actual_value: Gerçek değer
            
        Returns:
            bool: Doğru tahmin mi?
        """
        # Ağırlıklı oylama
        weighted_vote = 0
        total_weight = 0
        
        for name, (_, prob) in predictions.items():
            if name in self.weights:
                weighted_vote += prob * self.weights[name]
                total_weight += self.weights[name]
                
        if total_weight == 0:
            predicted_above = 0.5 >= 0.5
        else:
            predicted_above = (weighted_vote / total_weight) >= 0.5
            
        actual_above = actual_value >= self.threshold
        
        return predicted_above == actual_above
    
    def predict_next_value(self, sequence):
        """
        Bir sonraki değeri tahmin eder
        
        Args:
            sequence: Değerler dizisi
            
        Returns:
            tuple: (tahmini değer, eşik üstü olasılığı, tahminler sözlüğü)
        """
        if not self.models:
            return None, 0.5, {}
            
        # Her modelden tahmin al
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_next_value'):
                    result = model.predict_next_value(sequence)
                    
                    # Farklı tahmin metodlarını standartlaştır
                    if isinstance(result, tuple):
                        if len(result) >= 2:
                            pred, prob = result[0], result[1]
                        else:
                            pred, prob = result[0], 0.5 if pred is None else (1.0 if pred >= self.threshold else 0.0)
                    else:
                        pred = result
                        prob = 1.0 if pred >= self.threshold else 0.0
                        
                    predictions[name] = (pred, prob)
            except Exception as e:
                print(f"Model {name} tahmin hatası: {e}")
        
        if not predictions:
            return None, 0.5, {}
            
        # Değer tahmini (ağırlıklı ortalama)
        value_sum = 0
        value_weight = 0
        
        for name, (pred, _) in predictions.items():
            if pred is not None and name in self.weights:
                value_sum += pred * self.weights[name]
                value_weight += self.weights[name]
                
        prediction = value_sum / value_weight if value_weight > 0 else None
        
        # Eşik üstü olasılığı (ağırlıklı oylama)
        prob_sum = 0
        prob_weight = 0
        
        for name, (_, prob) in predictions.items():
            if name in self.weights:
                prob_sum += prob * self.weights[name]
                prob_weight += self.weights[name]
                
        above_prob = prob_sum / prob_weight if prob_weight > 0 else 0.5
        
        return prediction, above_prob, predictions
    
    def get_model_weights(self):
        """
        Güncel model ağırlıklarını döndürür
        
        Returns:
            dict: Model ağırlıkları
        """
        return self.weights.copy()
    
    def get_performance_summary(self):
        """
        Performans özetini döndürür
        
        Returns:
            pandas.DataFrame: Performans özeti
        """
        summary = []
        
        for name in self.models:
            global_stats = self.global_performance[name]
            global_acc = global_stats['correct'] / global_stats['total'] if global_stats['total'] > 0 else 0
            
            history = self.performance_history[name]
            recent_acc = sum(1 for x in history if x) / len(history) if history else 0
            
            summary.append({
                'Model': name,
                'Weight': self.weights[name],
                'Recent Accuracy': recent_acc,
                'Global Accuracy': global_acc,
                'Total Predictions': global_stats['total']
            })
            
        return pd.DataFrame(summary)
    
    def plot_performance(self, figsize=(12, 6)):
        """
        Performans grafiğini çizer
        
        Args:
            figsize: Grafik boyutu
            
        Returns:
            matplotlib.figure.Figure: Grafik
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Doğruluk grafiği
            summary = self.get_performance_summary()
            summary.plot(x='Model', y=['Recent Accuracy', 'Global Accuracy'], 
                         kind='bar', ax=ax1)
            ax1.set_title('Model Accuracy')
            ax1.set_ylim(0, 1)
            ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            
            # Ağırlık grafiği
            summary.plot(x='Model', y='Weight', kind='bar', ax=ax2)
            ax2.set_title('Model Weights')
            ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            return fig
