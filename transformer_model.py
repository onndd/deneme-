import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    GlobalAveragePooling1D, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

class TransformerModel:
    def __init__(self, seq_length=200, n_features=1, threshold=1.5):
        """
        Transformer modeli
        
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
        
    def build_model(self, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, 
                   mlp_units=[128], dropout=0.1, mlp_dropout=0.2, learning_rate=0.001):
        """
        Transformer modelini oluşturur
        
        Args:
            head_size: Her attention head'in boyutu
            num_heads: Attention head sayısı
            ff_dim: Feed-forward boyutu
            num_transformer_blocks: Transformer blok sayısı
            mlp_units: MLP katmanı boyutları
            dropout: Transformer dropout oranı
            mlp_dropout: MLP dropout oranı
            learning_rate: Öğrenme oranı
        """
        inputs = Input(shape=(self.seq_length, self.n_features))
        x = inputs
        
        # Transformer blokları
        for _ in range(num_transformer_blocks):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
            
            # Skip connection 1
            x = LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed-forward network
            ffn_output = Dense(ff_dim, activation="relu")(x)
            ffn_output = Dense(self.n_features)(ffn_output)
            
            # Skip connection 2
            x = LayerNormalization(epsilon=1e-6)(ffn_output + x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # MLP kafası
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)
            
        # Çıkış katmanı
        if self.is_binary:
            outputs = Dense(1, activation="sigmoid")(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            outputs = Dense(1, activation="linear")(x)
            loss = 'mse'
            metrics = ['mae']
        
        # Model oluştur
        model = Model(inputs, outputs)
        
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
