import numpy as np
import pandas as pd

# Kategori tanımları
# ÖNERİLEN YENİ VALUE_CATEGORIES
VALUE_CATEGORIES = {
    # Crash Bölgesi (1.00 - 1.09) - Çok detaylı
    'CRASH_100_101': (1.00, 1.01),  # Sadece 1.00x
    'CRASH_101_103': (1.01, 1.03),
    'CRASH_103_106': (1.03, 1.06),
    'CRASH_106_110': (1.06, 1.10),

    # Düşük Bölge (1.10 - 1.49) - Detaylı
    'LOW_110_115': (1.10, 1.15),
    'LOW_115_120': (1.15, 1.20),
    'LOW_120_125': (1.20, 1.25),
    'LOW_125_130': (1.25, 1.30),
    'LOW_130_135': (1.30, 1.35),
    'LOW_135_140': (1.35, 1.40),
    'LOW_140_145': (1.40, 1.45),
    'LOW_145_149': (1.45, 1.49), # 1.5 eşiğine en yakın alt bölge

    # Eşik Bölgesi (1.50 - 1.99) - Orta Detaylı
    'THRESH_150_160': (1.50, 1.60), # 1.5 eşiğini hemen geçenler
    'THRESH_160_170': (1.60, 1.70),
    'THRESH_170_185': (1.70, 1.85),
    'THRESH_185_199': (1.85, 1.99), # 2x'e yakın

    # Erken Çarpanlar (2.00 - 4.99)
    'EARLY_2X': (2.00, 2.49),
    'EARLY_2_5X': (2.50, 2.99),
    'EARLY_3X': (3.00, 3.99),
    'EARLY_4X': (4.00, 4.99),

    # Orta Çarpanlar (5.00 - 19.99)
    'MID_5_7X': (5.00, 7.49),
    'MID_7_10X': (7.50, 9.99),
    'MID_10_15X': (10.00, 14.99),
    'MID_15_20X': (15.00, 19.99),

    # Yüksek Çarpanlar (20.00 - 99.99)
    'HIGH_20_30X': (20.00, 29.99),
    'HIGH_30_50X': (30.00, 49.99),
    'HIGH_50_70X': (50.00, 69.99),
    'HIGH_70_100X': (70.00, 99.99),

    # Çok Yüksek Çarpanlar (100.00+)
    'XHIGH_100_200X': (100.00, 199.99),
    'XHIGH_200PLUS': (200.00, float('inf')) # En yüksek kategori
}


STEP_CATEGORIES = {
    'T1': (1.00, 1.49),
    'T2': (1.50, 2.00),
    'T3': (2.00, 2.50),
    'T4': (2.50, 3.00),
    'T5': (3.00, 3.50),
    'T6': (3.50, 4.00),
    'T7': (4.00, 4.50),
    'T8': (4.50, 5.00),
    'T9': (5.00, float('inf'))
}

SEQUENCE_CATEGORIES = {
    'S1': (5, 10),
    'S2': (10, 20),
    'S3': (20, 50),
    'S4': (50, 100),
    'S5': (100, 200),
    'S6': (200, 500),
    'S7': (500, 1000),
    'S8': (1000, float('inf'))
}

def get_value_category(value):
    """
    Değerin detaylı kategorisini döndürür.
    
    Args:
        value: JetX oyun sonucu (katsayı)
        
    Returns:
        str: Kategori kodu
    """
    for category, (min_val, max_val) in VALUE_CATEGORIES.items():
        if min_val <= value < max_val:
            return category
    return 'XHIGH_200PLUS'  # Eğer hiçbir kategoriye girmezse en yüksek kategori

def get_step_category(value):
    """
    0.5 adımlı değer kategorisini döndürür (T1-T9)
    
    Args:
        value: JetX oyun sonucu (katsayı)
        
    Returns:
        str: Kategori kodu
    """
    for category, (min_val, max_val) in STEP_CATEGORIES.items():
        if min_val <= value < max_val:
            return category
    return 'T9'  # Eğer hiçbir kategoriye girmezse en yüksek kategori

def get_sequence_category(seq_length):
    """
    Sıra uzunluğu kategorisini döndürür (S1-S8)
    
    Args:
        seq_length: Dizi uzunluğu
        
    Returns:
        str: Kategori kodu
    """
    for category, (min_val, max_val) in SEQUENCE_CATEGORIES.items():
        if min_val <= seq_length < max_val:
            return category
    return 'S8'  # Eğer hiçbir kategoriye girmezse en yüksek kategori

def get_compound_category(value, seq_length):
    """
    Çaprazlamalı kategori oluşturur (VALUE_CATEGORIES, STEP_CATEGORIES, SEQUENCE_CATEGORIES kullanarak)
    
    Args:
        value: JetX oyun sonucu (katsayı)
        seq_length: Dizi uzunluğu
        
    Returns:
        str: Çaprazlamalı kategori kodu
    """
    val_cat = get_value_category(value) # Yeni detaylı VALUE_CATEGORIES'i kullanır
    step_cat = get_step_category(value)
    seq_cat = get_sequence_category(seq_length)
    
    return f"{val_cat}__{step_cat}__{seq_cat}" # Ayraçları daha belirgin yaptım

def transform_to_categories(values):
    """
    Değerleri detaylı kategorilere dönüştürür.
    
    Args:
        values: JetX değerlerinin listesi
        
    Returns:
        list: Detaylı kategori kodlarının listesi
    """
    return [get_value_category(val) for val in values]

def transform_to_step_categories(values):
    """
    Değerleri 0.5 adımlı kategorilere dönüştürür
    
    Args:
        values: JetX değerlerinin listesi
        
    Returns:
        list: Kategori kodlarının listesi (T1-T9)
    """
    return [get_step_category(val) for val in values]

def transform_to_compound_categories(values):
    """
    Değerleri çaprazlamalı kategorilere dönüştürür (VALUE, STEP, SEQUENCE kullanarak)
    
    Args:
        values: JetX değerlerinin listesi
        
    Returns:
        list: Çaprazlamalı kategori kodlarının listesi
    """
    result = []
    for i, val in enumerate(values):
        seq_length = len(values) - i  # Geriye kalan eleman sayısı
        result.append(get_compound_category(val, seq_length))
    return result

def fuzzy_membership(value, category_key):
    """
    Bir değerin belirli bir VALUE_CATEGORIES kategorisine üyelik derecesini hesaplar (0-1 arası).
    
    Args:
        value: JetX oyun sonucu (katsayı)
        category_key: VALUE_CATEGORIES içindeki kategori anahtarı (örn: 'LOW_110_115')
        
    Returns:
        float: Üyelik derecesi (0-1 arası)
    """
    if category_key not in VALUE_CATEGORIES:
        # Eğer bilinmeyen bir kategori anahtarı gelirse, üyelik 0
        # Bu durumun oluşmaması gerekir, çağıran kodun geçerli anahtarlar kullanması beklenir.
        return 0.0

    min_val, max_val = VALUE_CATEGORIES[category_key]
    
    # float('inf') içeren aralıklar için özel durum
    if max_val == float('inf'):
        if value >= min_val:
            return 1.0 # Sonsuz aralığın içindeyse tam üyelik
        # Sonsuz aralığın alt sınırına yakınlık için bir mantık eklenebilir,
        # ancak şimdilik bu aralığın dışındaysa 0 döndürelim.
        # Veya bir önceki sonlu aralığa olan komşuluğuna bakılabilir.
        # Basitlik adına, şimdilik sadece aralık içindeyse 1, değilse 0.
        # Daha sofistike bir bulanıklaştırma için bu kısım geliştirilebilir.
        # Örneğin, min_val'den ne kadar uzaklaştığına bağlı olarak azalan bir üyelik.
        # Şimdilik, eğer değer min_val'den küçükse, bir önceki kategoriye olan komşuluğuna bakalım.
        if value < min_val: 
             # Bir önceki kategoriyi bulup ona olan komşuluğu hesaplamaya çalışabiliriz,
             # ancak bu fonksiyonun karmaşıklığını artırır.
             # Şimdilik, sonsuz aralığın altındaysa 0 kabul edelim.
             pass # Aşağıdaki genel komşuluk mantığına düşebilir.


    range_size = max_val - min_val if max_val != float('inf') else (min_val * 0.2 if min_val > 0 else 1.0) # Sonsuz aralık için varsayılan bir "genişlik"
    if range_size <= 0: range_size = 0.1 # Bölme hatasını önle

    # Eğer değer tam olarak bu aralıktaysa
    if min_val <= value < max_val:
        if max_val == float('inf'): # Sonsuz aralık için
            return 1.0
        # Aralığın ortasına yakınlık ile üyelik derecesini artır
        mid_point = (min_val + max_val) / 2
        distance_to_mid = abs(value - mid_point)
        max_distance_to_mid = range_size / 2
        if max_distance_to_mid == 0: return 1.0 # Tek noktalı aralık
        
        # Merkeze yakınlık derecesi (1 = tam merkezde, 0.5 = sınırda)
        return 1.0 - (distance_to_mid / max_distance_to_mid) * 0.5
    
    # Komşu kategorilere kısmi üyelik (Bu kısım daha genel hale getirilebilir)
    # Şu anki haliyle, sadece verilen 'category_key' dışındaki diğer tüm kategorilere bakar.
    # Daha iyi bir yaklaşım, gerçekten "komşu" olan kategorileri bulmak olabilir.
    # Şimdilik, değerin başka bir kategorinin sınırlarına ne kadar yakın olduğuna bakalım.
    # Bu fonksiyonun ana amacı, bir değerin *belirtilen kategoriye* olan üyeliği olduğu için,
    # komşuluk mantığı burada çok detaylı olmayabilir.
    # Eğer değer belirtilen aralığın dışındaysa, genellikle 0 döndürülür.
    # Ancak, bulanıklaştırma için hafif bir "taşma" payı verilebilir.
    
    # Değerin, sorgulanan kategorinin sınırlarına olan uzaklığına göre küçük bir üyelik verelim
    # (eğer hemen dışındaysa)
    overlap_ratio = 0.1 # Sınırların %10'u kadar dışına taşabilir
    extended_min = min_val - range_size * overlap_ratio
    extended_max = max_val + range_size * overlap_ratio if max_val != float('inf') else float('inf')

    if extended_min <= value < min_val: # Alt sınıra yakın ve dışında
        distance_to_boundary = min_val - value
        return max(0, 0.5 - (distance_to_boundary / (range_size * overlap_ratio)) * 0.5)
    if max_val != float('inf') and max_val <= value < extended_max: # Üst sınıra yakın ve dışında
        distance_to_boundary = value - max_val
        return max(0, 0.5 - (distance_to_boundary / (range_size * overlap_ratio)) * 0.5)
            
    return 0.0  # Bu kategoriye veya yakın komşuluğuna hiç üyelik yok

# YENİ EKLENEN FONKSİYONLAR
def get_value_step_crossed_category(value):
    """
    Yeni detaylı VALUE_CATEGORIES ile STEP_CATEGORIES'i çaprazlar.
    
    Args:
        value: JetX oyun sonucu (katsayı)
        
    Returns:
        str: Çaprazlanmış kategori kodu (örn: 'LOW_145_149__T1')
    """
    val_cat = get_value_category(value) # Yeni detaylı VALUE_CATEGORIES'i kullanır
    step_cat = get_step_category(value)
    
    return f"{val_cat}__{step_cat}"

def transform_to_value_step_crossed_categories(values):
    """
    Değer listesini yeni çapraz (VALUE ve STEP) kategorilere dönüştürür.
    
    Args:
        values: JetX değerlerinin listesi
        
    Returns:
        list: Çaprazlanmış kategori kodlarının listesi
    """
    return [get_value_step_crossed_category(val) for val in values]
