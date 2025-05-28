import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

def plot_value_distribution(values, threshold=1.5, bins=50, figsize=(10, 6)):
    """
    Değer dağılımını çizer

    Args:
        values: Değerler dizisi
        threshold: Eşik değeri
        bins: Histogram dilim sayısı
        figsize: Grafik boyutu

    Returns:
        matplotlib.figure.Figure: Grafik
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fig, ax = plt.subplots(figsize=figsize)

        # Eşik üstü/altı değerleri ayır
        below = [v for v in values if v < threshold]
        above = [v for v in values if v >= threshold]

        # Histogram
        if below:
            ax.hist(below, bins=bins, alpha=0.5, color='red', label=f'< {threshold}')
        if above:
            ax.hist(above, bins=bins, alpha=0.5, color='green', label=f'≥ {threshold}')

        # Eşik çizgisi
        ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')

        # İstatistikler
        mean_val = np.mean(values)
        median_val = np.median(values)

        ax.axvline(x=mean_val, color='blue', linestyle='-.', alpha=0.7, label=f'Mean ({mean_val:.2f})')
        ax.axvline(x=median_val, color='purple', linestyle=':', alpha=0.7, label=f'Median ({median_val:.2f})')

        # Eşik üstü oranı
        above_ratio = len(above) / len(values) if values else 0
        ax.text(0.02, 0.95, f'≥ {threshold} Ratio: {above_ratio:.3f}', transform=ax.transAxes,
               fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Value Distribution')
        ax.legend()

        return fig

def plot_category_distribution(values, threshold=1.5, figsize=(12, 6)):
    """
    Kategori dağılımını çizer

    Args:
        values: Değerler dizisi
        threshold: Eşik değeri (renk ayrımı için)
        figsize: Grafik boyutu

    Returns:
        matplotlib.figure.Figure: Grafik
    """
    from data_processing.transformer import transform_to_categories

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Kategorilere dönüştür
        categories = transform_to_categories(values)

        # Kategori sayıları
        category_counts = Counter(categories)

        # DataFrame'e dönüştür
        df = pd.DataFrame({'Category': list(category_counts.keys()),
                          'Count': list(category_counts.values())})

        # Kategorilere karşılık gelen ortalama değerleri hesapla
        from data_processing.transformer import VALUE_CATEGORIES

        category_values = {}
        for cat, count in category_counts.items():
            cat_values = [v for v, c in zip(values, categories) if c == cat]
            if cat_values:
                category_values[cat] = np.mean(cat_values)
            else:
                # Kategori orta noktası
                min_val, max_val = VALUE_CATEGORIES.get(cat, (1.0, 1.5))
                category_values[cat] = (min_val + max_val) / 2

        df['MeanValue'] = df['Category'].map(category_values)

        # Eşik üstü kategorileri belirle
        df['AboveThreshold'] = df['MeanValue'] >= threshold

        # Sıralama için kategori kodlarını ayır
        df['CategoryType'] = df['Category'].str[0]
        df['CategoryNumber'] = df['Category'].str[1:].astype(int)

        # Kategori tipine ve numarasına göre sırala
        df = df.sort_values(['CategoryType', 'CategoryNumber'])

        # Grafik
        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(df['Category'], df['Count'], 
                     color=df['AboveThreshold'].map({True: 'green', False: 'red'}),
                     alpha=0.7)

        # Değer etiketleri
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_title('Category Distribution')

        # Her kategori için mean değeri
        ax2 = ax.twinx()
        ax2.plot(df['Category'], df['MeanValue'], 'bo-', alpha=0.6)
        ax2.set_ylabel('Mean Value', color='blue')

        # Eşik çizgisi
        ax2.axhline(y=threshold, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

def plot_transition_matrix(values, figsize=(10, 8)):
    """
    Geçiş matrisini görselleştirir

    Args:
        values: Değerler dizisi
        figsize: Grafik boyutu

    Returns:
        matplotlib.figure.Figure: Grafik
    """
    from data_processing.transformer import transform_to_categories
    from models.transition.transition_matrix import TransitionMatrixModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Kategorilere dönüştür
        categories = transform_to_categories(values)

        # Geçiş matrisini oluştur
        tm_model = TransitionMatrixModel(use_categories=True)
        tm_model.fit(values)

        # Geçiş matrisini DataFrame'e dönüştür
        transition_df = tm_model.get_transition_matrix(as_dataframe=True)

        if transition_df.empty:
            return None

        # Grafik
        fig, ax = plt.subplots(figsize=figsize)

        # Isı haritası
        sns.heatmap(transition_df, annot=True, cmap='Blues', fmt='.2f', ax=ax)

        ax.set_xlabel('Next Category')
        ax.set_ylabel('Current Category')
        ax.set_title('Transition Matrix')

        plt.tight_layout()
        return fig

def plot_time_patterns(values, timestamps=None, threshold=1.5, figsize=(12, 8)):
    """
    Zamansal örüntüleri görselleştirir

    Args:
        values: Değerler dizisi
        timestamps: Zaman damgaları (None = otomatik oluştur)
        threshold: Eşik değeri
        figsize: Grafik boyutu

    Returns:
        matplotlib.figure.Figure: Grafik
    """
    if not values:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Zaman damgaları yoksa otomatik oluştur
        if timestamps is None:
            from datetime import datetime, timedelta

            # Şu anki zamandan geriye doğru 1'er dakika aralıklarla oluştur
            end_time = datetime.now()
            timestamps = [end_time - timedelta(minutes=i) for i in range(len(values))]
            timestamps.reverse()  # En eski zaman başta

        # DataFrame oluştur
        df = pd.DataFrame({'value': values, 'timestamp': pd.to_datetime(timestamps)})

        # Aynı kodun geri kalanı...

        # Zaman özellikleri
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Pazartesi, 6=Pazar
        df['is_weekend'] = df['day_of_week'] >= 5  # Hafta sonu mu?
        df['above_threshold'] = df['value'] >= threshold  # Eşik üstü mü?

        # Grafik
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # Saat bazında analiz
        hour_df = df.groupby('hour').agg(
            mean_value=('value', 'mean'),
            above_ratio=('above_threshold', 'mean'),
            count=('value', 'count')
        ).reset_index()

        ax1 = axes[0]

        # Ortalama değer
        ax1.plot(hour_df['hour'], hour_df['mean_value'], 'bo-', label='Mean Value')

        # Eşik üstü oranı
        ax12 = ax1.twinx()
        ax12.plot(hour_df['hour'], hour_df['above_ratio'], 'ro-', label='Above Ratio')

        # Eşik çizgisi
        ax1.axhline(y=threshold, color='black', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')

        # Etiketler
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Mean Value', color='blue')
        ax12.set_ylabel('Above Threshold Ratio', color='red')
        ax1.set_title('Value by Hour of Day')

        # X ekseni tam sayılar
        ax1.set_xticks(range(0, 24, 2))

        # Histogram
        ax1.bar(hour_df['hour'], hour_df['count'] / hour_df['count'].max() * threshold/2,
               alpha=0.2, color='gray', label='Sample Count (scaled)')

        # Efsaneler
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax12.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        # Gün bazında analiz
        day_df = df.groupby('day_of_week').agg(
            mean_value=('value', 'mean'),
            above_ratio=('above_threshold', 'mean'),
            count=('value', 'count')
        ).reset_index()

        ax2 = axes[1]

        # Ortalama değer
        ax2.plot(day_df['day_of_week'], day_df['mean_value'], 'bo-', label='Mean Value')

        # Eşik üstü oranı
        ax22 = ax2.twinx()
        ax22.plot(day_df['day_of_week'], day_df['above_ratio'], 'ro-', label='Above Ratio')

        # Eşik çizgisi
        ax2.axhline(y=threshold, color='black', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')

        # Etiketler
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Mean Value', color='blue')
        ax22.set_ylabel('Above Threshold Ratio', color='red')
        ax2.set_title('Value by Day of Week')

        # X ekseni tam sayılar ve günler
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(days)

        # Hafta içi/sonu belirt
        ax2.axvspan(5, 7, alpha=0.2, color='yellow', label='Weekend')

        # Histogram
        ax2.bar(day_df['day_of_week'], day_df['count'] / day_df['count'].max() * threshold/2,
               alpha=0.2, color='gray', label='Sample Count (scaled)')

        # Efsaneler
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax22.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.tight_layout()
        return fig
