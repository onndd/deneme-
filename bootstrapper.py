import numpy as np
import pandas as pd
from datetime import datetime, time

def block_bootstrap(data, block_size=20, num_samples=100):
    """
    Blok bootstrap örnekleme yapar

    Args:
        data: Veri listesi veya dizisi
        block_size: Her bir bloğun büyüklüğü
        num_samples: Oluşturulacak örnek sayısı

    Returns:
        list: Bootstrap örneklemlerinin listesi
    """
    data = np.array(data)
    n = len(data)
    samples = []

    for _ in range(num_samples):
        # Rastgele başlangıç noktaları seç
        num_blocks = int(np.ceil(n / block_size))
        # En fazla n - block_size + 1 başlangıç noktası olabilir
        max_start = max(1, n - block_size + 1)
        start_indices = np.random.randint(0, max_start, num_blocks)

        # Blokları birleştir
        bootstrap_sample = []
        for start in start_indices:
            end = min(start + block_size, n)
            bootstrap_sample.extend(data[start:end])

            # Yeterince veri toplandıysa döngüden çık
            if len(bootstrap_sample) >= n:
                bootstrap_sample = bootstrap_sample[:n]
                break

        samples.append(bootstrap_sample)

    return samples

def weighted_bootstrap(data, weights=None, block_size=20, num_samples=100):
    """
    Ağırlıklı bootstrap örnekleme yapar

    Args:
        data: Veri listesi veya dizisi
        weights: Veri puanlarının ağırlıkları (None = son 1000 %60, gerisi %40)
        block_size: Her bir bloğun büyüklüğü
        num_samples: Oluşturulacak örnek sayısı

    Returns:
        list: Bootstrap örneklemlerinin listesi
    """
    data = np.array(data)
    n = len(data)

    # Ağırlıkları tanımla (verilen değilse son 1000 %60, gerisi %40)
    if weights is None:
        weights = np.ones(n)
        if n > 1000:
            weights[-1000:] = 1.5  # Son 1000 örneğe %50 daha fazla ağırlık

    # Ağırlıkları normalize et
    weights = weights / np.sum(weights)

    samples = []
    for _ in range(num_samples):
        # Ağırlıklı olarak blok başlangıç noktaları seç
        num_blocks = int(np.ceil(n / block_size))
        sample_indices = []

        for _ in range(num_blocks):
            # Ağırlıklı örnekleme
            start = np.random.choice(range(n), p=weights)

            # Bloğu ekle
            end = min(start + block_size, n)
            block = list(range(start, end))
            sample_indices.extend(block)

            # Yeterince veri toplandıysa döngüden çık
            if len(sample_indices) >= n:
                sample_indices = sample_indices[:n]
                break

        # İndeksleri benzersiz yap ve sırala
        sample_indices = sorted(list(set(sample_indices)))

        # İndekslere göre örnekleme yap
        bootstrap_sample = data[sample_indices]
        samples.append(bootstrap_sample)

    return samples

def time_based_bootstrap(data, timestamps=None, block_size=20, num_samples=100):
    """
    Zaman dilimine göre bootstrap örnekleme yapar, ancak zaman verisi yoksa normal block bootstrap kullanır

    Args:
        data: Veri listesi veya dizisi
        timestamps: Zaman damgaları (kullanılmıyor)
        block_size: Her bir bloğun büyüklüğü
        num_samples: Oluşturulacak örnek sayısı

    Returns:
        list: Bootstrap örneklemlerinin listesi
    """
    # Zaman verisi olmadığı için normal block bootstrap kullan
    return block_bootstrap(data, block_size, num_samples)
