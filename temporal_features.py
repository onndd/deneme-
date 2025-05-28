import numpy as np

def extract_time_features(timestamps):
    """
    Zaman verisi olmadığı için boş dizi döndürür
    """
    return np.array([])

def calculate_time_differences(timestamps):
    """
    Zaman verisi olmadığı için boş dizi döndürür
    """
    return np.array([])

def combine_temporal_features(values, timestamps=None):
    """
    Zaman verisi olmadığı için boş özellik matrisi döndürür
    """
    return np.zeros((len(values), 0))
