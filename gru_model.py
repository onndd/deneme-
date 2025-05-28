import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

class GRUModel:
    def __init__(self, seq_length=200, n_features=1, threshold=1.5):
        """
        GRU modeli
        
        Args:
            seq_length: Dizi uzunluğu
            n_features: Özellik sayısı
            threshold: Eşik değeri
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.threshold = threshold
        self.model = None
        self.history = None
        self.is_binary = True  # İkili sınıflandırma (eşik üstü/altı)
    
    def build_model(self, gru_units=64, dropout_rate=0.2, learning_rate=0.001):
        """
        GRU modelini oluşturur
        
        Args:
            gru_units: GRU birim sayısı
            dropout_rate: Dropout oranı
            learning_rate: Öğrenme oranı
        """
        model = Sequential()
        
        # GRU katmanları
        model.add(GRU(gru_units, return_sequences=True, 
                      input_shape=(self.seq_length, self.n_features)))
        model.add(Dropout(dropout_rate))
        
        model.add(GRU(gru_units // 2, return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        # Çıkış katmanı
        if self.is_binary:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Modeli derle
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model = model
        return model
    
    def prepare_sequences(self, data):
        """
        Eğitim dizilerini hazırlar
        
        Args:
            data: Veri dizisi
            
        Returns:
            tuple: X (girdi dizileri) ve y (hedef değerler)
        """
        X, y = [], []
        
        for i in range(len(data) - self.seq_length):
            # Dizi
            seq = data[i:i+self.seq_length]
            
            # Sonraki değer
            next_val = data[i+self.seq_length]
            
            X.append(seq)
            if self.is_binary:
                y.append(1 if next_val >= self.threshold else 0)
            else:
                y.append(next_val)
        
        return np.array(X), np.array(y)
    
    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """
        Modeli eğitir
        
        Args:
            X: Girdi dizileri
            y: Hedef değerler
            epochs: Epoch sayısı
            batch_size: Batch boyutu
            validation_split: Doğrulama seti oranı
            verbose: Çıktı detay seviyesi
            
        Returns:
            history: Eğitim geçmişi
        """
        # Erken durdurma
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Modeli eğit
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        self.history = history
        return history
    
    def predict(self, X):
        """
        Tahmin yapar
        
        Args:
            X: Girdi dizisi
            
        Returns:
            numpy.ndarray: Tahminler
        """
        return self.model.predict(X)
    
    def predict_next(self, sequence):
        """
        Bir sonraki değeri tahmin eder
        
        Args:
            sequence: Girdi dizisi
            
        Returns:
            tuple: (tahmin değeri, eşik üstü olasılığı)
        """
        # Boyut kontrolü
        if len(sequence) < self.seq_length:
            # Yeterli veri yoksa, mevcut dizi ile doldur
            padding = [sequence[0]] * (self.seq_length - len(sequence))
            sequence = padding + list(sequence)
        
        # Son seq_length değeri al
        sequence = sequence[-self.seq_length:]
        
        # Yeniden şekillendir
        X = np.array(sequence).reshape(1, self.seq_length, self.n_features)
        
        # Tahmin
        prediction = self.model.predict(X)[0][0]
        
        if self.is_binary:
            # İkili sınıflandırma: Eşik üstü olasılığı
            return None, prediction
        else:
            # Regresyon: Tahmin değeri ve eşik üstü olma durumu
            above_threshold = prediction >= self.threshold
            return prediction, 1.0 if above_threshold else 0.0
    
    def evaluate(self, X, y):
        """
        Model performansını değerlendirir
        
        Args:
            X: Test girdileri
            y: Test hedefleri
            
        Returns:
            tuple: (loss, accuracy) veya (loss, mae)
        """
        return self.model.evaluate(X, y)
    
    def save(self, filepath):
        """
        Modeli kaydeder
        
        Args:
            filepath: Dosya yolu
        """
        self.model.save(filepath)
    
    def load(self, filepath):
        """
        Modeli yükler
        
        Args:
            filepath: Dosya yolu
        """
        self.model = tf.keras.models.load_model(filepath)
