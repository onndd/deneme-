import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from .metrics import calculate_threshold_metrics
import matplotlib.pyplot as plt
import warnings

def time_series_cv(predictor, data, n_splits=5, test_size=0.1, threshold=1.5, window_size=100):
    """
    Zaman serisi çapraz doğrulama yapar
    
    Args:
        predictor: Tahmin modeli
        data: Veri
        n_splits: Bölüm sayısı
        test_size: Test seti oranı
        threshold: Eşik değeri
        window_size: Tahmin penceresi boyutu
        
    Returns:
        pandas.DataFrame: CV sonuçları
    """
    # TimeSeriesSplit ile bölümleri oluştur
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(data) * test_size))
    
    results = []
    
    for i, (train_index, test_index) in enumerate(tscv.split(data)):
        # Eğitim/test verilerini ayır
        train_data = data[train_index]
        test_data = data[test_index]
        
        # Modeli eğit
        if hasattr(predictor, 'fit'):
            predictor.fit(train_data)
        
        # Test verileri üzerinde tahmin yap
        predictions = []
        actuals = []
        
        for j in range(window_size, len(test_data)):
            # Tahmin penceresi
            history = test_data[max(0, j-window_size):j]
            
            # Gerçek sonraki değer
            actual = test_data[j]
            
            try:
                # Tahmin yap
                result = predictor.predict_next_value(history)
                
                if isinstance(result, tuple) and len(result) >= 2:
                    above_prob = result[1]
                elif isinstance(result, (int, float)):
                    above_prob = 1.0 if result >= threshold else 0.0
                else:
                    continue
                
                # Sonuçları kaydet
                predictions.append(1.0 if above_prob >= 0.5 else 0.0)
                actuals.append(actual)
            except:
                continue
        
        # Metrikleri hesapla
        if predictions:
            metrics = calculate_threshold_metrics(actuals, predictions, threshold=threshold)
            
            results.append({
                'Fold': i + 1,
                'Train Size': len(train_data),
                'Test Size': len(test_data),
                'Accuracy': metrics['accuracy'],
                'Balanced Accuracy': metrics['balanced_accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'Above Ratio': sum(1 for a in actuals if a >= threshold) / len(actuals)
            })
    
    # DataFrame'e dönüştür
    return pd.DataFrame(results)

def plot_cv_results(cv_results, figsize=(12, 6)):
    """
    Çapraz doğrulama sonuçlarını görselleştirir
    
    Args:
        cv_results: CV sonuçları DataFrame
        figsize: Grafik boyutu
        
    Returns:
        matplotlib.figure.Figure: Grafik
    """
    if cv_results.empty:
        return None
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Metrikler
        metrics = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1']
        
        # Grafik renkleri
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        # Her metrik için çizgi
        for i, metric in enumerate(metrics):
            ax.plot(cv_results['Fold'], cv_results[metric], f'o-', color=colors[i], label=metric)
            
        # Ortalama çizgileri
        for i, metric in enumerate(metrics):
            mean_value = cv_results[metric].mean()
            ax.axhline(y=mean_value, color=colors[i], linestyle='--', alpha=0.5,
                      label=f'Mean {metric}: {mean_value:.3f}')
        
        # Şans çizgisi
        ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='Chance Level')
        
        # Above ratio (veri dengesizliği)
        above_ratio = cv_results['Above Ratio'].mean()
        ax.axhline(y=above_ratio, color='gray', linestyle='-.', alpha=0.5,
                  label=f'Above 1.5 Ratio: {above_ratio:.3f}')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_title('Cross-Validation Results')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # X ekseni tam sayılar
        ax.set_xticks(cv_results['Fold'])
        
        plt.tight_layout()
        return fig
