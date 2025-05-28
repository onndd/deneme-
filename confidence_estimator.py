import numpy as np
from scipy import stats

class ConfidenceEstimator:
    def __init__(self, history_size=100):
        """
        Güven Skoru Tahmincisi

        Args:
            history_size: Geçmiş tahmin sayısı
        """
        self.history_size = history_size
        self.prediction_history = []
        self.actual_history = []
        self.confidence_history = []

    def add_prediction(self, prediction, actual_value, confidence):
        """
        Yeni bir tahmin-sonuç çiftini kaydeder

        Args:
            prediction: Tahmin edilen değer
            actual_value: Gerçek değer
            confidence: Tahmin güveni
        """
        self.prediction_history.append(prediction)
        self.actual_history.append(actual_value)
        self.confidence_history.append(confidence)

        # Geçmiş boyutunu sınırla
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
            self.actual_history.pop(0)
            self.confidence_history.pop(0)

    def calculate_calibration_score(self):
        """
        Güven kalibrasyonu skorunu hesaplar

        Returns:
            float: Kalibrasyon skoru
        """
        # --- YENİ KONTROLLER ---
        # None ve NaN olmayan geçerli güven skorlarını al
        valid_confidences_in_history = [
            c for c in self.confidence_history 
            if c is not None and not (isinstance(c, float) and np.isnan(c))
        ]

        if len(valid_confidences_in_history) < 10: # Yeterli geçmiş veri yoksa varsayılan kalibrasyon
            return 0.5 
            
        if len(self.prediction_history) < 10: # Bu kontrol de kalabilir
            return 0.5

        # Tahminlerin ikili dönüşümü (eşik üstü/altı)
        binary_predictions = [1 if p >= 1.5 else 0 for p in self.prediction_history] # p None olmamalı
        binary_actuals = [1 if a >= 1.5 else 0 for a in self.actual_history]

        # Doğruluk
        accuracy = sum(1 for p, a in zip(binary_predictions, binary_actuals) if p == a) / len(binary_predictions)

        # Ortalama güven
        mean_confidence = np.mean(valid_confidences_in_history) # Sadece geçerli olanların ortalaması

        # Kalibrasyon skoru (1 = tam kalibre, 0 = hiç kalibre değil)
        calibration = 1.0 - abs(accuracy - mean_confidence)

        return calibration

    def estimate_confidence(self, prediction, above_prob, uncertainty=None):
        """
        Tahmin için güven skorunu tahmin eder

        Args:
            prediction: Tahmin edilen değer
            above_prob: Eşik üstü olasılığı
            uncertainty: Tahmin belirsizliği (opsiyonel)

        Returns:
            float: Güven skoru (0-1 arası)
        """
        if not self.prediction_history:
            # Geçmiş yok, varsayılan güven
            return 0.5

        # Geçmiş tahminlere göre belirsizlik
        prediction_std = np.std(self.prediction_history)
        if prediction_std == 0:
            prediction_std = 0.1  # Sıfır bölme hatası

        # Tahminin geçmiş tahminlere olan mesafesi
        z_score = 0
        if prediction is not None:
            z_score = abs(prediction - np.mean(self.prediction_history)) / prediction_std

        # Z-skoru güven skoruna dönüştür (1 = çok yakın, 0 = çok uzak)
        z_confidence = 1.0 / (1.0 + z_score)
        
         # --- YENİ KONTROL ---
        current_above_prob = above_prob
        if current_above_prob is None or (isinstance(current_above_prob, float) and np.isnan(current_above_prob)):
            # print("estimate_confidence: above_prob None veya NaN geldi, 0.5 varsayılıyor.")
            current_above_prob = 0.5

        # Kesinlik faktörü (ne kadar kesin tahmin, o kadar güvenilir)
        certainty = 1.0 - 2.0 * abs(current_above_prob - 0.5)  # 0.5 = belirsiz, 0/1 = kesin

        # Hesaplanmış belirsizlik varsa kullan
        if uncertainty is not None:
            certainty *= (1.0 - min(1.0, uncertainty))

        # Kalibrasyon
        calibration = self.calculate_calibration_score()

        # Nihai güven skoru (ağırlıklı ortalama)
        confidence = 0.4 * z_confidence + 0.4 * certainty + 0.2 * calibration

        return confidence
