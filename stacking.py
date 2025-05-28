import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

class StackingEnsemble:
    def __init__(self, models=None, meta_model=None, threshold=1.5, is_classifier=True):
        """
        Stacking Ensemble model
        
        Args:
            models: Alt modeller sözlüğü
            meta_model: Üst model (None = otomatik seçim)
            threshold: Eşik değeri
            is_classifier: Sınıflandırma mı?
        """
        self.models = models or {}
        self.meta_model = meta_model
        self.threshold = threshold
        self.is_classifier = is_classifier
        
        # Otomatik model seçimi
        if self.meta_model is None:
            if self.is_classifier:
                self.meta_model = LogisticRegression(C=0.1, class_weight='balanced')
            else:
                self.meta_model = LinearRegression()
        
        # Eğitim verileri
        self.X_meta = []
        self.y_meta = []
        
        # Model aktif mi?
        self.is_fitted = False
    
    def add_model(self, name, model):
        """
        Yeni bir model ekler
        
        Args:
            name: Model adı
            model: Model nesnesi
        """
        self.models[name] = model
        self.is_fitted = False
    
    def collect_predictions(self, sequence):
        """
        Alt modellerin tahminlerini toplar
        
        Args:
            sequence: Değerler dizisi
            
        Returns:
            numpy.ndarray: Tahmin matrisi
        """
        predictions = []
        
        for name, model in self.models.items():
            try:
                # Modelin tahmin metodunu çağır
                if hasattr(model, 'predict_next_value'):
                    result = model.predict_next_value(sequence)
                    
                    # Farklı tahmin metodlarını standartlaştır
                    if isinstance(result, tuple) and len(result) >= 2:
                        pred, prob = result[0], result[1]
                        
                        # Tahmin ve olasılık
                        if pred is not None:
                            predictions.append(pred)
                        predictions.append(prob)
                        
                        # Tahmini dönüştür
                        if self.is_classifier:
                            predictions.append(1.0 if prob >= 0.5 else 0.0)
                            
                    elif isinstance(result, (int, float)):
                        # Sadece tahmin
                        predictions.append(result)
                        
                        # Tahmini dönüştür
                        if self.is_classifier:
                            predictions.append(1.0 if result >= self.threshold else 0.0)
            except Exception as e:
                print(f"Model {name} tahmin hatası: {e}")
                # Eksik değerleri doldur
                predictions.extend([0.5, 0.5] if self.is_classifier else [self.threshold])
        
        return np.array(predictions).reshape(1, -1)
    
    def add_training_example(self, sequence, actual_value):
        """
        Eğitim örneği ekler
        
        Args:
            sequence: Değerler dizisi
            actual_value: Gerçek değer
        """
        # Alt model tahminlerini topla
        X = self.collect_predictions(sequence)
        
        # Hedef değeri dönüştür
        if self.is_classifier:
            y = 1 if actual_value >= self.threshold else 0
        else:
            y = actual_value
            
        # Eğitim verisine ekle
        self.X_meta.append(X[0])
        self.y_meta.append(y)
    
    def fit_meta_model(self):
        """
        Üst modeli eğitir
        """
        if not self.X_meta or not self.y_meta:
            return False
            
        # Verileri numpy dizisine dönüştür
        X = np.array(self.X_meta)
        y = np.array(self.y_meta)
        
        # Uyarıları bastır
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                # Modeli eğit
                self.meta_model.fit(X, y)
                self.is_fitted = True
                return True
            except Exception as e:
                print(f"Meta model eğitim hatası: {e}")
                return False
    
    def predict_next_value(self, sequence):
        """
        Bir sonraki değeri tahmin eder
        
        Args:
            sequence: Değerler dizisi
            
        Returns:
            tuple: (tahmini değer, eşik üstü olasılığı)
        """
        # Alt model tahminlerini topla
        X = self.collect_predictions(sequence)
        
        # Meta model eğitilmiş mi?
        if not self.is_fitted or len(self.X_meta) < 10:
            # Modeli eğitmeye çalış
            if len(self.X_meta) >= 10:
                self.fit_meta_model()
                
            # Meta model yoksa, ortalama tahmin
            if not self.is_fitted:
                # Alt model tahminlerinin ortalaması
                if self.is_classifier:
                    # 3 elemanlı gruplar: [değer, olasılık, sınıf]
                    probs = X[0][1::3]
                    above_prob = np.mean(probs)
                    return None, above_prob
                else:
                    # Değer tahminleri
                    preds = X[0][::2]
                    valid_preds = [p for p in preds if p is not None]
                    if not valid_preds:
                        return None, 0.5
                        
                    prediction = np.mean(valid_preds)
                    above_prob = 1.0 if prediction >= self.threshold else 0.0
                    return prediction, above_prob
        
        # Meta model ile tahmin
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                if self.is_classifier:
                    # Sınıf olasılığı
                    if hasattr(self.meta_model, 'predict_proba'):
                        proba = self.meta_model.predict_proba(X)[0]
                        above_prob = proba[1] if len(proba) > 1 else 0.5
                    else:
                        pred = self.meta_model.predict(X)[0]
                        above_prob = 1.0 if pred >= 0.5 else 0.0
                        
                    # Alt modellerden değer tahminini al
                    predictions = [X[0][i] for i in range(0, len(X[0]), 3)]
                    valid_preds = [p for p in predictions if p is not None]
                    
                    if valid_preds:
                        # Ağırlıklı ortalama (olasılık ile)
                        probs = [X[0][i+1] for i in range(0, len(X[0]), 3) if X[0][i] is not None]
                        weights = [abs(p - 0.5) + 0.1 for p in probs]  # 0.5'ten uzaklık
                        prediction = np.average(valid_preds, weights=weights)
                    else:
                        prediction = None
                        
                    return prediction, above_prob
                else:
                    # Doğrudan değer tahmini
                    prediction = self.meta_model.predict(X)[0]
                    
                    # Eşik kontrolü
                    above_prob = 1.0 if prediction >= self.threshold else 0.0
                    
                    return prediction, above_prob
            except Exception as e:
                print(f"Meta model tahmin hatası: {e}")
                return None, 0.5
