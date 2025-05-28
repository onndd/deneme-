import sqlite3
import pandas as pd
import numpy as np

def load_data_from_sqlite(db_path="jetx_data.db", limit=None):
    """
    SQLite veritabanından JetX verilerini yükler
    
    Args:
        db_path: SQLite veritabanı dosya yolu (.db uzantılı)
        limit: Yüklenecek son kayıt sayısı (None=tümü)
    
    Returns:
        pandas.DataFrame: Yüklenen veriler
    """
    conn = sqlite3.connect(db_path)
    
    # Tablo yoksa oluştur
    conn.execute('''
    CREATE TABLE IF NOT EXISTS jetx_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        value REAL NOT NULL
    )
    ''')
    
    # Predictions tablosu da oluştur
    conn.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predicted_value REAL,
        confidence_score REAL,
        above_threshold INTEGER,
        actual_value REAL,
        was_correct INTEGER
    )
    ''')
    
    # Verileri çek
    if limit:
        query = f"SELECT * FROM jetx_results ORDER BY id DESC LIMIT {limit}"
    else:
        query = "SELECT * FROM jetx_results ORDER BY id"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def save_result_to_sqlite(value, db_path="jetx_data.db"):
    """
    Yeni bir JetX sonucunu veritabanına kaydeder
    
    Args:
        value: JetX oyun sonucu (katsayı)
        db_path: SQLite veritabanı dosya yolu
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO jetx_results (value) VALUES (?)
    ''', (value,))
    
    conn.commit()
    conn.close()
    
    return cursor.lastrowid

def save_prediction_to_sqlite(prediction_data, db_path="jetx_data.db"):
    """
    Tahmin sonuçlarını SQLite veritabanına kaydeder
    
    Args:
        prediction_data: Kaydedilecek tahmin verisi (dict)
        db_path: SQLite veritabanı dosya yolu
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO predictions 
    (predicted_value, confidence_score, above_threshold)
    VALUES (?, ?, ?)
    ''', (
        prediction_data['predicted_value'],
        prediction_data['confidence_score'],
        1 if prediction_data['above_threshold'] else 0
    ))
    
    conn.commit()
    conn.close()
    
    return cursor.lastrowid

def update_prediction_result(prediction_id, actual_value, db_path="jetx_data.db"):
    """
    Tahmin sonucunu günceller (gerçek değer öğrenildiğinde)
    
    Args:
        prediction_id: Güncellenecek tahmin ID'si
        actual_value: Gerçekleşen JetX değeri
        db_path: SQLite veritabanı dosya yolu
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Önce mevcut tahmini çek
    cursor.execute("SELECT above_threshold FROM predictions WHERE id=?", (prediction_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False
    
    above_threshold = row[0]
    was_correct = 1 if (above_threshold == 1 and actual_value >= 1.5) or \
                       (above_threshold == 0 and actual_value < 1.5) else 0
    
    # Tahmini güncelle
    cursor.execute('''
    UPDATE predictions 
    SET actual_value=?, was_correct=? 
    WHERE id=?
    ''', (actual_value, was_correct, prediction_id))
    
    conn.commit()
    conn.close()
    
    return True
