import numpy as np

class CrashDetector:
    def __init__(self, crash_threshold=1.5, lookback_period=5,
                 very_low_value_threshold=1.05, low_avg_threshold=1.20,
                 sustained_high_streak_threshold=1.80, streak_length_for_sustained_high=4):
        """
        1.5 altı (crash) durumlarını tespit etmeye odaklanmış özel model.

        Args:
            crash_threshold (float): 1.5 gibi, crash olarak kabul edilecek eşik.
            lookback_period (int): Özellikleri hesaplamak için bakılacak son el sayısı.
            very_low_value_threshold (float): Bir değerin "çok düşük" kabul edileceği eşik (örn: 1.05).
            low_avg_threshold (float): Son X elin ortalamasının "düşük" kabul edileceği eşik (örn: 1.20).
            sustained_high_streak_threshold (float): Değerlerin "yüksekte seyrediyor" kabul edileceği eşik (örn: 1.80).
            streak_length_for_sustained_high (int): Yüksekte seyretme serisinin alarm vermesi için gereken minimum uzunluk.
        """
        self.crash_threshold = crash_threshold
        self.lookback_period = lookback_period
        self.very_low_value_threshold = very_low_value_threshold
        self.low_avg_threshold = low_avg_threshold
        self.sustained_high_streak_threshold = sustained_high_streak_threshold
        self.streak_length_for_sustained_high = streak_length_for_sustained_high

        self.historical_crash_precursor_features = [] # Gelecekte öğrenme için kullanılabilir
        print(f"CrashDetector başlatıldı: Geriye bakış={self.lookback_period}, Çok Düşük Eşik={self.very_low_value_threshold}, Düşük Ort. Eşik={self.low_avg_threshold}")

    def _extract_features(self, sequence):
        """
        Verilen bir diziden crash habercisi özellikleri çıkarır.
        Dizi, en az self.lookback_period uzunluğunda olmalıdır.
        """
        if len(sequence) < self.lookback_period:
            return {
                'avg_last_n': None,
                'count_very_low': 0,
                'all_above_sustained_high': False,
                'is_decreasing_sharply': False
            }

        relevant_sequence = sequence[-self.lookback_period:]

        avg_last_n = np.mean(relevant_sequence)
        count_very_low = sum(1 for x in relevant_sequence if x < self.very_low_value_threshold)
        all_above_sustained_high = all(x > self.sustained_high_streak_threshold for x in relevant_sequence)

        is_decreasing_sharply = False
        if len(relevant_sequence) >= 3: # Keskin düşüş için en az 3 değere bakalım
            # Son 3 değerin sürekli ve belirgin düştüğünü kontrol et (basit bir yaklaşım)
            if relevant_sequence[-1] < relevant_sequence[-2] < relevant_sequence[-3] and \
               (relevant_sequence[-3] - relevant_sequence[-1]) > 0.3: # Örnek: 0.3'ten fazla bir düşüş
                is_decreasing_sharply = True

        return {
            'avg_last_n': avg_last_n,
            'count_very_low': count_very_low,
            'all_above_sustained_high': all_above_sustained_high,
            'is_decreasing_sharply': is_decreasing_sharply
        }

    def fit(self, historical_values):
        """
        Modeli geçmiş verilere göre eğitir/ayarlar.
        Bu basit versiyonda, sadece geçmiş crash'lerden önce gelen özellikleri toplayabiliriz.
        Daha karmaşık bir model burada eğitilebilirdi (örn: Logistic Regression).
        """
        print(f"CrashDetector {len(historical_values)} değer ile 'fit' ediliyor (basit özellik toplama)...")
        self.historical_crash_precursor_features = []
        if len(historical_values) <= self.lookback_period:
            print("CrashDetector fit için yeterli geçmiş veri yok.")
            return

        for i in range(self.lookback_period, len(historical_values)):
            current_value = historical_values[i]
            if current_value < self.crash_threshold: # Eğer bir crash olduysa
                # Crash'ten hemen önceki diziyi al
                precursor_sequence = historical_values[i - self.lookback_period : i]
                features = self._extract_features(precursor_sequence)
                self.historical_crash_precursor_features.append(features)

        print(f"{len(self.historical_crash_precursor_features)} adet crash öncesi özellik seti toplandı.")
        # Burada bu özelliklerin istatistikleri çıkarılabilir veya basit bir model eğitilebilir.
        # Şimdilik sadece topluyoruz.

    def predict_crash_risk(self, current_sequence):
        """
        Verilen mevcut diziye göre bir crash risk skoru (0-1 arası) tahmin eder.
        1'e yakın skor, yüksek crash riski anlamına gelir.
        """
        if len(current_sequence) < self.lookback_period:
            return 0.0 # Yeterli geçmiş yoksa düşük risk

        features = self._extract_features(current_sequence)
        risk_score = 0.0

        if features['avg_last_n'] is None: # Özellik hesaplanamadıysa
            return 0.0

        # Basit Kural Tabanlı Risk Değerlendirmesi (Bu kurallar ve ağırlıklar deneme yanılma ile geliştirilebilir)
        if features['avg_last_n'] < self.low_avg_threshold:
            risk_score += 0.4

        if features['count_very_low'] >= 1: # Son X elde en az 1 tane çok düşük değer varsa
            risk_score += 0.3
        if features['count_very_low'] >= 2: # Son X elde 2 veya daha fazla çok düşük değer varsa
            risk_score += 0.2 # Ekstra risk

        if features['all_above_sustained_high'] and len(current_sequence) >= self.streak_length_for_sustained_high:
             # Eğer son 'streak_length_for_sustained_high' eldir hep yüksek geliyorsa (bir "düzeltme" beklentisi)
            risk_score += 0.25 

        if features['is_decreasing_sharply']:
            risk_score += 0.35

        # Risk skorunu 0 ile 1 arasında sınırla
        return min(1.0, risk_score)