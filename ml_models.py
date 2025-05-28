import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

class RandomForestJetXPredictor:
    def __init__(self, n_estimators=100, random_state=42, threshold=1.5, n_lags=10):
        """
        Random Forest Tabanlı JetX Tahmincisi

        Args:
            n_estimators: Ormandaki ağaç sayısı.
            random_state: Rastgelelik için tohum değeri.
            threshold: 1.5 üstü/altı karar sınırı.
            n_lags: Tahmin için kullanılacak geçmiş değer sayısı (özellik sayısı).
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.threshold = threshold
        self.n_lags = n_lags
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight='balanced' # 1.5 altı durumları daha iyi yakalamak için önemli olabilir
        )
        self.is_fitted = False
        print(f"RandomForestJetXPredictor başlatıldı: n_lags={self.n_lags}, n_estimators={self.n_estimators}")

    def _create_features(self, values):
        """
        Verilen değerler dizisinden gecikmeli özellikleri (lagged features) oluşturur.
        """
        X, y = [], []
        if len(values) <= self.n_lags:
            return np.array(X), np.array(y)

        for i in range(len(values) - self.n_lags):
            # Son n_lags değeri özellik olarak al
            feature_sequence = values[i : i + self.n_lags]
            X.append(feature_sequence)
            # Bir sonraki değeri hedef olarak al (1.5 üstü mü, altı mı?)
            target_value = values[i + self.n_lags]
            y.append(1 if target_value >= self.threshold else 0)
        return np.array(X), np.array(y)

    def fit(self, values):
        """
        Modeli verilen JetX değerleriyle eğitir.

        Args:
            values: Eğitim için kullanılacak geçmiş JetX değerleri listesi/dizisi.
        """
        print(f"RandomForestJetXPredictor {len(values)} değer ile eğitime başlıyor...")
        if len(values) <= self.n_lags:
            print(f"RandomForest için yeterli veri yok (en az {self.n_lags + 1} değer gerekli). Eğitim atlanıyor.")
            self.is_fitted = False
            return

        X_train, y_train = self._create_features(values)

        if X_train.shape[0] == 0:
            print("RandomForest için özellik çıkarılamadı. Eğitim atlanıyor.")
            self.is_fitted = False
            return

        try:
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            print("RandomForestJetXPredictor başarıyla eğitildi.")
        except Exception as e:
            print(f"RandomForest eğitimi sırasında hata: {e}")
            self.is_fitted = False

    def predict_next_value(self, sequence):
        """
        Verilen son değerler dizisine (sequence) göre bir sonraki adım için tahmin yapar.

        Args:
            sequence: Son JetX değerlerini içeren liste/dizi.

        Returns:
            tuple: (None, 1.5_ustu_olasiligi, guven_skoru)
                   RandomForestClassifier doğrudan değer tahmini yapmaz, olasılık verir.
                   Güven skoru için basit bir yer tutucu kullanılır.
        """
        if not self.is_fitted:
            # print("RF Modeli henüz eğitilmemiş. Varsayılan tahmin (0.5 olasılık).")
            return None, 0.5, 0.3 # Eğitilmemişse düşük güvenle ortada bir tahmin

        if len(sequence) < self.n_lags:
            # print(f"RF Tahmini için yeterli geçmiş veri yok (en az {self.n_lags} gerekli). Varsayılan tahmin.")
            return None, 0.5, 0.3

        # Tahmin için son n_lags değeri özellik olarak al
        current_features = np.array(sequence[-self.n_lags:]).reshape(1, -1)

        try:
            # Olasılıkları tahmin et [P(0_olasiligi), P(1_olasiligi)]
            probabilities = self.model.predict_proba(current_features)[0]
            above_threshold_probability = probabilities[1] # 1.5 üstü olma olasılığı (sınıf 1)

            # Güven skoru için basit bir yaklaşım: Olasılığın 0.5'ten ne kadar uzak olduğu
            confidence = abs(above_threshold_probability - 0.5) * 2 
            # Bu, olasılık 0 veya 1 ise güven 1; olasılık 0.5 ise güven 0 olur.

            # RandomForestClassifier doğrudan bir "değer" tahmini yapmaz, sınıf olasılığı verir.
            # Değer tahmini için None döndürüyoruz. Ensemble bunu dikkate alacaktır.
            return None, above_threshold_probability, confidence
        except NotFittedError:
            # print("RF Modeli (predict_next_value içinde) eğitilmemiş. Varsayılan tahmin.")
            return None, 0.5, 0.3
        except Exception as e:
            print(f"RandomForest tahmini sırasında hata: {e}")
            return None, 0.5, 0.3