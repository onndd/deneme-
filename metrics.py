import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

def calculate_threshold_metrics(y_true, y_pred, threshold=1.5):
    """
    Eşik tabanlı sınıflandırma metrikleri hesaplar
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin değerleri
        threshold: Eşik değeri
        
    Returns:
        dict: Metrikler
    """
    # İkili sınıflara dönüştür
    y_true_binary = np.array(y_true) >= threshold
    y_pred_binary = np.array(y_pred) >= threshold
    
    # Accuracy
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Dengeli doğruluk
    balanced_accuracy = (recall + (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp
    }

def calculate_regression_metrics(y_true, y_pred):
    """
    Regresyon metrikleri hesaplar
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin değerleri
        
    Returns:
        dict: Metrikler
    """
    # Array'e dönüştür
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if (y_true != 0).all() else np.inf
    
    # R^2
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

def plot_confusion_matrix(y_true, y_pred, threshold=1.5, figsize=(8, 6)):
    """
    Karmaşıklık matrisini çizer
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin değerleri
        threshold: Eşik değeri
        figsize: Grafik boyutu
        
    Returns:
        matplotlib.figure.Figure: Grafik
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # İkili sınıflara dönüştür
        y_true_binary = np.array(y_true) >= threshold
        y_pred_binary = np.array(y_pred) >= threshold
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        # Normalize et
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Grafik
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['< 1.5', '≥ 1.5'], yticklabels=['< 1.5', '≥ 1.5'])
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        return fig

def plot_prediction_vs_actual(y_true, y_pred, figsize=(10, 6)):
    """
    Tahmin vs gerçek değer grafiğini çizer
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin değerleri
        figsize: Grafik boyutu
        
    Returns:
        matplotlib.figure.Figure: Grafik
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Mükemmel tahmin çizgisi
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Eşik çizgileri
        ax.axhline(y=1.5, color='g', linestyle='--', alpha=0.5)
        ax.axvline(x=1.5, color='g', linestyle='--', alpha=0.5)
        
        # Hata bölgeleri
        y_true_binary = np.array(y_true) >= 1.5
        y_pred_binary = np.array(y_pred) >= 1.5
        
        for i in range(len(y_true)):
            if y_true_binary[i] != y_pred_binary[i]:
                ax.plot(y_true[i], y_pred[i], 'ro', markersize=8, mfc='none')
        
        ax.set_xlabel('Actual Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title('Prediction vs Actual')
        
        return fig

def plot_accuracy_by_confidence(predictions, actuals, confidences, bins=10, figsize=(10, 6)):
    """
    Güven skoruna göre doğruluk grafiğini çizer
    
    Args:
        predictions: Tahminler (eşik üstü olasılıkları)
        actuals: Gerçek değerler
        confidences: Güven skorları
        bins: Dilim sayısı
        figsize: Grafik boyutu
        
    Returns:
        matplotlib.figure.Figure: Grafik
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Veri hazırlama
        data = []
        
        for pred, actual, conf in zip(predictions, actuals, confidences):
            # İkili dönüşüm
            pred_binary = pred >= 0.5
            actual_binary = actual >= 1.5
            
            # Doğruluk
            correct = pred_binary == actual_binary
            
            data.append({'Confidence': conf, 'Correct': correct})
            
        df = pd.DataFrame(data)
        
        # Güven aralıklarına göre grupla
        df['Confidence Bin'] = pd.cut(df['Confidence'], bins=bins)
        
        # Ortalama doğruluk
        accuracy_by_conf = df.groupby('Confidence Bin')['Correct'].mean().reset_index()
        counts_by_conf = df.groupby('Confidence Bin').size().reset_index(name='Count')
        
        merged = pd.merge(accuracy_by_conf, counts_by_conf, on='Confidence Bin')
        
        # Grafik
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(range(len(merged)), merged['Correct'], alpha=0.7)
        
        # Bar etiketleri
        ax.set_xticks(range(len(merged)))
        ax.set_xticklabels([f"{interval.left:.1f}-{interval.right:.1f}" for interval in merged['Confidence Bin']], 
                          rotation=45, ha='right')
        
        # Sayı etiketleri
        for i, bar in enumerate(bars):
            count = merged.iloc[i]['Count']
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"n={count}", ha='center', va='bottom', fontsize=8)
            
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Confidence Score')
        ax.set_ylim(0, 1.1)
        
        # Şans çizgisi
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
