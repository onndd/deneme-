import numpy as np
from collections import defaultdict

class WeightedEnsemble:
    def __init__(self, models=None, weights=None, threshold=1.5):
        """
        Ağırlıklı Ensemble modeli

        Args:
            models: Model nesnelerinin sözlüğü
            weights: Model ağırlıklarının sözlüğü
            threshold: Eşik değeri
        """
        self.models = models or {}
        self.weights = weights or {}
        self.threshold = threshold

        # Performans istatistikleri
        self.performance = defaultdict(lambda: {'correct': 0, 'total': 0})

        # Tüm modeller için eşit başlangıç ağırlığı
        for model_name in self.models:
            if model_name not in self.weights:
                self.weights[model_name] = 1.0

    def add_model(self, name, model, weight=1.0):
        """
        Yeni bir model ekler

        Args:
            name: Model adı
            model: Model nesnesi
            weight: Model ağırlığı
        """
        self.models[name] = model
        self.weights[name] = weight

    def predict_next_value(self, sequence):
        """
        Bir sonraki değeri tahmin eder

        Args:
            sequence: Değerler dizisi

        Returns:
            tuple: (tahmini değer, eşik üstü olasılığı, güven skoru)
        """
        if not self.models:
            return None, 0.5, 0.0

        predictions = {}
        above_probs = {}
        confidences = {}

        # Her modelden tahmin al
        for name, model in self.models.items():
            try:
                # Modelin tahmin metodunu çağır
                if hasattr(model, 'predict_next_value'):
                    result = model.predict_next_value(sequence)

                    # Farklı tahmin metodlarını standartlaştır
                    if len(result) == 2:
                        pred, prob = result
                        conf = 0.5  # Varsayılan güven
                    elif len(result) == 3:
                        pred, prob, conf = result
                    else:
                        continue

                    predictions[name] = pred
                    above_probs[name] = prob
                    confidences[name] = conf
            except Exception as e:
                print(f"Model {name} tahmin hatası: {e}")

        if not predictions:
            return None, 0.5, 0.0

        # Tahmini değerleri ağırlıklı olarak birleştir
        value_predictions = []
        value_weights = []

        for name, pred in predictions.items():
            if pred is not None:
                # Değer tahmini için ağırlık (model ağırlığı * güven)
                weight = self.weights[name] * confidences.get(name, 0.5)
                value_predictions.append(pred)
                value_weights.append(weight)

        if not value_predictions:
            value_prediction = None
        else:
            total_weight = sum(value_weights)
            if total_weight > 0:
                # Ağırlıklı ortalama
                value_prediction = sum(p * w for p, w in zip(value_predictions, value_weights)) / total_weight
            else:
                value_prediction = np.mean(value_predictions)

        # Eşik üstü olasılığını ağırlıklı olarak birleştir
        prob_weights = []
        prob_predictions = []

        for name, prob in above_probs.items():
            # Olasılık tahmini için ağırlık
            weight = self.weights[name] * confidences.get(name, 0.5)
            prob_predictions.append(prob)
            prob_weights.append(weight)

        total_weight = sum(prob_weights)
        if total_weight > 0:
            # Ağırlıklı ortalama
            above_prob = sum(p * w for p, w in zip(prob_predictions, prob_weights)) / total_weight
        else:
            above_prob = np.mean(prob_predictions)

        # Genel güven skoru
        confidence = np.mean(list(confidences.values())) if confidences else 0.5

        return value_prediction, above_prob, confidence

    # weighted_average.py -> WeightedEnsemble sınıfı içinde
    def update_weights(self, correct_predictions, actual_value, threshold): # <<<--- YENİ PARAMETRELER EKLENDİ
        """
        Model ağırlıklarını günceller.
        1.5 altı durumları doğru bilmeye veya yanlış bilmeye farklı ağırlıklar uygular.

        Args:
            correct_predictions: Doğru tahmin yapan modellerin listesi
            actual_value: Gerçekleşen JetX değeri
            threshold: Karar eşiği (örn: 1.5)
        """

        # Bu değerleri deneyerek ayarlayabilirsiniz:
        reward_boost_for_below_1_5_correct = 1.5  # 1.5 altını doğru bilirse normalden %50 daha fazla ödül
        penalty_multiplier_for_below_1_5_incorrect = 1.6 # 1.5 altını yanlış bilirse normalden %50 daha fazla ceza

        # Mevcut ağırlıkların ne kadar korunacağı, yeni performansın ne kadar etki edeceği
        smoothing_factor_old_weight = 0.8 # Eski ağırlığın %80'i korunur
        smoothing_factor_new_performance = 0.2 # Yeni performansın %20'si etki eder

        is_critical_case = (actual_value < threshold) # Bu el "1.5 altı" mıydı?

        for name in self.models: # Tüm modeller için (sadece self.models'a eklenmiş olanlar)
            if name not in self.performance: # Eğer model için performans kaydı yoksa başlat
                 self.performance[name] = {'correct': 0, 'total': 0}

            self.performance[name]['total'] += 1
            is_model_correct = (name in correct_predictions)

            current_accuracy = 0.0
            if is_model_correct:
                self.performance[name]['correct'] += 1
                current_accuracy = 1.0 # Model doğru bildi
                if is_critical_case: # Eğer kritik bir "1.5 altı" durumu doğru bildiyse
                    current_accuracy *= reward_boost_for_below_1_5_correct # Ödülünü artır
            else:
                current_accuracy = 0.0 # Model yanlış bildi
                if is_critical_case: # Eğer kritik bir "1.5 altı" durumu YANLIŞ bildiyse
                                     # (yani 1.5 üstü dedi ama 1.5 altı geldi)
                                     # Daha fazla cezalandır (negatif bir etki gibi düşünülebilir)
                    current_accuracy = -1.0 * penalty_multiplier_for_below_1_5_incorrect # Negatif performans gibi düşünün
                                                                                         # veya ağırlığı daha fazla azaltın


            # Ağırlığı yumuşak bir şekilde güncelle
            # Eğer current_accuracy negatif ise (kritik hatada), bu ağırlığı azaltacaktır.
            # Pozitif ve boost edilmişse, daha fazla artıracaktır.
            new_calculated_weight_effect = smoothing_factor_new_performance * current_accuracy

            # Eğer current_accuracy negatif ise (yani kritik bir hata yapılmışsa),
            # ağırlığı direkt olarak bu negatif etki kadar azaltmak yerine,
            # mevcut ağırlığı daha fazla düşürmeyi veya daha az artırmayı düşünebiliriz.
            # Daha basit bir yaklaşım için, doğruluğu 0 ve 1 arasında tutalım ama kritik durumlarda
            # güncelleme adımını daha etkili yapalım.

            # Yeniden düzenlenmiş ağırlık güncelleme mantığı:
            # 1. Modelin genel doğruluğunu hesapla
            stats = self.performance[name]
            overall_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.5

            # 2. Kritik durumlar için ayarlama yap
            adjustment_factor = 1.0
            if is_critical_case:
                if is_model_correct:
                    adjustment_factor = reward_boost_for_below_1_5_correct
                else: # Kritik durumu yanlış bildi
                    adjustment_factor = 1.0 / penalty_multiplier_for_below_1_5_incorrect # Etkisini azalt (1'den küçük)

            # 3. Nihai ağırlığı hesapla
            # Önceki ağırlığın bir kısmını koru, yeni (ayarlanmış) doğruluğun bir kısmını ekle
            # Ayarlanmış doğruluk, genel doğruluk * ayarlama faktörü olabilir.
            adjusted_accuracy_contribution = overall_accuracy * adjustment_factor

            self.weights[name] = (smoothing_factor_old_weight * self.weights.get(name, 1.0)) + \
                                 (smoothing_factor_new_performance * adjusted_accuracy_contribution)

            # Ağırlıkların negatif olmamasını ve makul bir aralıkta kalmasını sağlayalım
            if self.weights[name] < 0.01: # Çok küçük bir pozitif değer
                self.weights[name] = 0.01

        # Ağırlıkları normalize et (toplamları 1 olacak şekilde) - bu önemli
        total_weight_sum = sum(self.weights.values())
        if total_weight_sum > 0:
            for name_to_normalize in self.weights:
                self.weights[name_to_normalize] /= total_weight_sum
        else: # Eğer tüm ağırlıklar sıfırlanırsa (çok olası değil ama)
            for name_to_normalize in self.weights:
                self.weights[name_to_normalize] = 1.0 / len(self.weights) if self.weights else 1.0

    def get_model_info(self):
        """
        Model bilgilerini döndürür

        Returns:
            dict: Model bilgileri
        """
        info = {}
        for name in self.models:
            stats = self.performance[name]
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

            info[name] = {
                'weight': self.weights[name],
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }

        return info
