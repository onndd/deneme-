import numpy as np
from collections import defaultdict, Counter

class MarkovModel:
    def __init__(self, order=1, threshold=1.5, use_categories=True):
        """
        Markov Zinciri modeli

        Args:
            order: Markov zinciri derecesi
            threshold: Eşik değeri
            use_categories: Kategoriler kullanılsın mı?
        """
        self.order = order
        self.threshold = threshold
        self.use_categories = use_categories

        # Geçiş matrisleri
        self.transitions = defaultdict(Counter)
        self.threshold_transitions = defaultdict(Counter)

        # İstatistikler
        self.total_transitions = defaultdict(int)
        self.total_threshold_transitions = defaultdict(int)

    def _get_state(self, values):
        """
        Değerler dizisinden durum oluşturur

        Args:
            values: Değerler listesi

        Returns:
            tuple: Durum
        """
        if self.use_categories:
            from data_processing.transformer import transform_to_categories
            return tuple(transform_to_categories(values))
        else:
            # Ondalıkları yuvarla (hash için)
            return tuple(round(x, 2) for x in values)

    def _is_above_threshold(self, value):
        """
        Değerin eşik üstünde olup olmadığını kontrol eder

        Args:
            value: Değer

        Returns:
            bool: Eşik üstünde mi?
        """
        # Eğer value bir string ise, sayıya dönüştürmeye çalış
        if isinstance(value, str):
            from data_processing.transformer import VALUE_CATEGORIES
            min_val, max_val = VALUE_CATEGORIES.get(value, (1.0, 1.5))
            value = (min_val + max_val) / 2

        return float(value) >= self.threshold

    def fit(self, values):
        """
        Modeli veriye göre eğitir

        Args:
            values: Değerler dizisi
        """
        # Geçişleri hesapla
        for i in range(len(values) - self.order):
            # Geçerli durum
            current_state = self._get_state(values[i:i+self.order])

            # Sonraki değer
            next_value = values[i+self.order]

            # Sonraki durum (kategorik veya sayısal)
            if self.use_categories:
                from data_processing.transformer import get_value_category
                next_state = get_value_category(next_value)
            else:
                next_state = round(next_value, 2)

            # Geçişleri güncelle
            self.transitions[current_state][next_state] += 1
            self.total_transitions[current_state] += 1

            # Eşik geçişlerini güncelle
            is_above = self._is_above_threshold(next_value)
            self.threshold_transitions[current_state][is_above] += 1
            self.total_threshold_transitions[current_state] += 1

    def predict_proba(self, sequence):
        """
        Durum için olasılıkları tahmin eder

        Args:
            sequence: Değerler dizisi

        Returns:
            tuple: (durum olasılıkları, eşik üstü olasılığı)
        """
        # Durum oluştur
        if len(sequence) < self.order:
            # Veri yetersizse tahmin yapılamaz
            return {}, 0.5

        # Son n değeri al
        sequence = sequence[-self.order:]
        state = self._get_state(sequence)

        # Bu durum daha önce görülmemiş
        if state not in self.transitions or self.total_transitions[state] == 0:
            # Genel istatistiklere dayalı tahmin yap
            all_values = []
            for s, counts in self.transitions.items():
                for next_state, count in counts.items():
                    all_values.extend([next_state] * count)

            if not all_values:
                return {}, 0.5

            # Genel olasılıkları hesapla
            state_counter = Counter(all_values)
            total = sum(state_counter.values())
            state_probs = {state: count/total for state, count in state_counter.items()}

            # Eşik üstü olasılığı
            above_count = 0
            for val in all_values:
                # Kategori mi sayı mı kontrol et
                if isinstance(val, str):
                    from data_processing.transformer import VALUE_CATEGORIES
                    min_val, max_val = VALUE_CATEGORIES.get(val, (1.0, 1.5))
                    val_numeric = (min_val + max_val) / 2
                    if val_numeric >= self.threshold:
                        above_count += 1
                else:
                    if val >= self.threshold:
                        above_count += 1

            above_prob = above_count / total if total > 0 else 0.5

            return state_probs, above_prob

        # Durum olasılıklarını hesapla
        state_counts = self.transitions[state]
        total = self.total_transitions[state]
        state_probs = {next_state: count/total for next_state, count in state_counts.items()}

        # Eşik üstü olasılığını hesapla
        above_counts = self.threshold_transitions[state]
        above_total = self.total_threshold_transitions[state]
        above_prob = above_counts.get(True, 0) / above_total if above_total > 0 else 0.5

        return state_probs, above_prob

    def predict_next_value(self, sequence):
        """
        Bir sonraki değeri tahmin eder

        Args:
            sequence: Değerler dizisi

        Returns:
            tuple: (tahmini değer, eşik üstü olasılığı)
        """
        # Olasılıkları al
        state_probs, above_prob = self.predict_proba(sequence)

        if not state_probs:
            # Tahmin yok
            return None, above_prob

        # En olası durumu seç
        most_likely_state = max(state_probs, key=state_probs.get)

        if self.use_categories:
            # Kategori ortalamasını kullan
            from data_processing.transformer import VALUE_CATEGORIES
            min_val, max_val = VALUE_CATEGORIES.get(most_likely_state, (1.0, 1.5))
            prediction = (min_val + max_val) / 2
        else:
            # Doğrudan değeri kullan
            prediction = most_likely_state

        return prediction, above_prob
