import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import warnings
from models.ml_models import RandomForestJetXPredictor
import sqlite3
from datetime import datetime
import sys

# Uyarıları gizle
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow uyarılarını gizle

# main.py dosyasını manuel olarak içe aktaralım
# Eğer Colab'da sys.path'e proje ana dizinini eklediyseniz bu satır gerekmeyebilir
# veya __file__ Colab'da farklı çalışabileceği için dikkatli olunmalı.
# Genellikle Colab'da sys.path.append('/content/drive/MyDrive/jetxpredictor') gibi bir yol
# Hücre 2'de eklendiği için bu satır doğrudan çalışmayabilir veya gereksiz olabilir.
# Şimdilik yorumda bırakalım, Hücre 2'deki sys.path ayarı daha güvenilir.
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# JetXPredictor sınıfını direkt dosyadan al
# main.py'yi çalıştırma, sadece içindeki sınıfı kullan
from data_processing.loader import load_data_from_sqlite, save_result_to_sqlite, save_prediction_to_sqlite, update_prediction_result
from data_processing.transformer import transform_to_categories
from data_processing.splitter import create_sequences, create_above_threshold_target
from feature_engineering.categorical_features import extract_categorical_features
from feature_engineering.pattern_features import encode_pattern_features, extract_similarity_features
from feature_engineering.statistical_features import extract_statistical_features
#from models.sequential.lstm_model import LSTMModel # LSTM devre dışı olduğu için bu satır yorumlu kalabilir
from models.transition.markov_model import MarkovModel
from models.similarity.pattern_matcher import PatternMatcher # Güncellenmiş pattern_matcher.py kullanılacak
from models.statistical.density_model import DensityModel
from ensemble.weighted_average import WeightedEnsemble
from ensemble.confidence_estimator import ConfidenceEstimator
from models.crash_detector import CrashDetector

# JetXPredictor sınıfını manuel olarak tanımla
class JetXPredictor:
    def __init__(self, db_path="jetx_data.db"):
        """
        JetX Tahmin Sistemi
        Args:
            db_path: SQLite veritabanı dosya yolu
        """
        self.db_path = db_path
        self.models = {}
        self.ensemble = None
        self.confidence_estimator = ConfidenceEstimator()
        self.threshold = 1.5

        # Model özellikleri
        # self.lstm_seq_length = 100 # LSTM devre dışı olduğu için bu yorumlu kalabilir

        # MarkovModel Ayarları
        self.markov_order = 4 # Değiştirebileceğiniz yerlerden biri

        # --- Random Forest için Yeni Ayarlar ---
        self.rf_n_estimators = 90  # Ormandaki ağaç sayısı
        self.rf_n_lags = 15         # Geçmiş kaç değere bakılacağı (özellik sayısı)
        # -----------------------------------------

        # PatternMatcher Ayarları (MERKEZİ YÖNETİM İÇİN)
        self.pattern_lengths = [5, 8, 12, 15, 20,25, 30, 40, 75, 100, 175, 250] # Değiştirebileceğiniz yerlerden biri (Örnek değerler)
        self.pm_min_similarity_threshold = 0.75       # Değiştirebileceğiniz yerlerden biri (PatternMatcher için min benzerlik)
        self.pm_max_similar_patterns = 13             # Değiştirebileceğiniz yerlerden biri (PatternMatcher için maks. benzer örüntü)

        # --- CrashDetector için Yeni Ayarlar ---
        self.cd_lookback_period = 7       # CrashDetector son 7 ele bakacak
        self.cd_very_low_thresh = 1.05    # 1.05 altı değerler "çok düşük" sayılacak
        self.cd_low_avg_thresh = 1.25     # Son 7 elin ortalaması 1.25 altındaysa riskli
        self.cd_sust_high_thresh = 1.90   # Sürekli 1.90 üstü gelmesi durumu
        self.cd_sust_high_length = 5      # En az 5 el sürekli yüksek gelirse risk artar

        print("JetX Tahmin Sistemi başlatılıyor...")
        # CrashDetector'ı başlat
        self.crash_detector = CrashDetector(
            crash_threshold=self.threshold,
            lookback_period=self.cd_lookback_period,
            very_low_value_threshold=self.cd_very_low_thresh,
            low_avg_threshold=self.cd_low_avg_thresh,
            sustained_high_streak_threshold=self.cd_sust_high_thresh,
            streak_length_for_sustained_high=self.cd_sust_high_length
        )

    def load_data(self):
        """
        Veritabanından verileri yükler
        Returns:
            pandas.DataFrame: Yüklenen veriler
        """
        print(f"Veritabanından veriler yükleniyor: {self.db_path}")
        try:
            df = load_data_from_sqlite(self.db_path)
            print(f"Toplam {len(df)} veri noktası yüklendi.")
            return df
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return None

    def initialize_models(self, data):
        """
        Modelleri başlatır ve eğitir
        Args:
            data: Eğitim verisi
        """
        if data is None or len(data) == 0:
            print("Eğitim için veri bulunamadı!")
            return
        values = data['value'].values
        # Zaman verisi kullanmıyoruz
        print(f"Toplam {len(values)} veri noktası ile modeller eğitiliyor...")

        # LSTM modeli (Devre Dışı)
        print("LSTM modeli eğitimi atlandı (devre dışı).")

        # Markov modeli
        print("Markov modeli hazırlanıyor...")
        markov_model = MarkovModel(order=self.markov_order, threshold=self.threshold)
        markov_model.fit(values)

        # Örüntü eşleştirici
        print("Örüntü eşleştirici hazırlanıyor...")
        pattern_matcher = PatternMatcher(
            threshold=self.threshold,
            pattern_lengths=self.pattern_lengths, # JetXPredictor.__init__ içinden gelen değer
            min_similarity_threshold=self.pm_min_similarity_threshold, # JetXPredictor.__init__ içinden gelen değer
            max_similar_patterns=self.pm_max_similar_patterns      # JetXPredictor.__init__ içinden gelen değer
        )
        pattern_matcher.fit(values)

        # Yoğunluk modeli
        print("Yoğunluk modeli hazırlanıyor...")
        density_model = DensityModel(threshold=self.threshold)
        density_model.fit(values)

        # --- Random Forest Modelini Ekleme ---
        print("Random Forest modeli hazırlanıyor...")
        rf_model = RandomForestJetXPredictor(
            n_estimators=self.rf_n_estimators,
            random_state=42, # Sabit bir random_state ile sonuçların tekrarlanabilir olmasını sağlarız
            threshold=self.threshold,
            n_lags=self.rf_n_lags
        )
        rf_model.fit(values) # Tüm 'values' ile eğitilir
        # ------------------------------------

        # --- CrashDetector Modelini Eğitme/Hazırlama ---
        if hasattr(self, 'crash_detector') and self.crash_detector is not None:
            print("CrashDetector hazırlanıyor/fit ediliyor...")
            self.crash_detector.fit(values) # values = data['value'].values olmalı
            print("CrashDetector hazır.")
        # ---------------------------------------------

        # Modelleri kaydet (SADECE AKTİF MODELLER)
        # self.models sözlüğüne crash_detector'ı eklemiyoruz, çünkü o ensemble'a direkt katılmayacak,
        # çıktısını predict_next içinde ayrıca kullanacağız.
        self.models = {
            'markov': markov_model,
            'pattern_matcher': pattern_matcher,
            'density': density_model,
            'random_forest': rf_model 
        }

        self.ensemble = WeightedEnsemble(
            models=self.models,
            threshold=self.threshold
        )
        print("Tüm aktif modeller (LSTM hariç, Random Forest dahil) hazırlandı ve ensemble kuruldu!")

    def predict_next(self, recent_values=None):
        # ... (metodun başındaki recent_values alma kısmı aynı) ...
        # if len(recent_values) < 10: ... kısmı da aynı ...

        print(f"Son {len(recent_values)} değer kullanılarak tahmin yapılıyor...")

        if not self.ensemble:
            print("HATA: Ensemble modeli başlatılmamış.")
            return None

        # ADIM 1: Ana Ensemble Tahminini Al
        ensemble_value_pred, ensemble_above_prob, ensemble_confidence = self.ensemble.predict_next_value(recent_values)

        print(f"Ensemble Ham Tahmin Sonuçları: değer={ensemble_value_pred}, olasılık={ensemble_above_prob}, güven={ensemble_confidence}")

        # ADIM 2: CrashDetector'dan Risk Skoru Al
        crash_risk_score = 0.0 # Varsayılan
        if hasattr(self, 'crash_detector') and self.crash_detector is not None:
            # CrashDetector'ın kendi lookback_period'una yetecek kadar veri olduğundan emin ol
            if len(recent_values) >= self.cd_lookback_period: # self.cd_lookback_period __init__'te tanımlanmalı
                crash_risk_score = self.crash_detector.predict_crash_risk(recent_values)
                print(f"CrashDetector Risk Skoru: {crash_risk_score:.2f}")
            else:
                print(f"CrashDetector için yeterli geçmiş veri yok (en az {self.cd_lookback_period} gerekli), risk 0 kabul edildi.")

        # Güven skorunu (ensemble'ın güveni üzerinden) iyileştir
        final_confidence = self.confidence_estimator.estimate_confidence(ensemble_value_pred, ensemble_above_prob)

        # Karar Eşiği
        KARAR_ESIGI = 0.7 # Bunu 0.7 yapmıştınız, deneyerek değiştirebilirsiniz

        # ADIM 3: Nihai Kararı Ver (CrashDetector Riskini Kullanarak)
        final_above_threshold_decision = ensemble_above_prob >= KARAR_ESIGI if ensemble_above_prob is not None else False

        CRASH_DETECTOR_OVERRIDE_THRESHOLD = 0.7 # CrashDetector bu eşiğin üzerinde risk bildirirse, ensemble'ı ezer (deneyebilirsiniz)

        if crash_risk_score >= CRASH_DETECTOR_OVERRIDE_THRESHOLD:
            if final_above_threshold_decision: # Eğer ensemble 1.5 üstü diyorduysa
                print(f"UYARI: CrashDetector YÜKSEK RİSK ({crash_risk_score:.2f}) sinyali verdi! Ensemble tahmini ({ensemble_above_prob:.2f}) geçersiz kılınıp 1.5 altı tahmin ediliyor.")
            final_above_threshold_decision = False # CrashDetector yüksek risk verdiyse, 1.5 altı de.

        result = {
            'input_values': recent_values[-10:].tolist(), # Bu -10'u da bir değişkene bağlayabilirsiniz
            'predicted_value': ensemble_value_pred, # Ana ensemble'ın değer tahmini
            'above_threshold': final_above_threshold_decision, # CrashDetector tarafından etkilenmiş karar
            'above_threshold_probability': ensemble_above_prob, # Ana ensemble'ın olasılığı
            'confidence_score': final_confidence, # Genel güven
            'crash_risk_by_special_model': crash_risk_score # Özel modelin risk skoru
        }

        # Sonucu veritabanına kaydet
        prediction_data = {
            'predicted_value': ensemble_value_pred if ensemble_value_pred is not None else -1.0,
            'confidence_score': final_confidence if final_confidence is not None and not np.isnan(final_confidence) else 0.0,
            'above_threshold': final_above_threshold_decision,
            # İsterseniz crash_risk_score'u da veritabanına ekleyebilirsiniz,
            # bunun için predictions tablonuza yeni bir sütun eklemeniz gerekir.
        }
        try:
            prediction_id = save_prediction_to_sqlite(prediction_data, self.db_path)
            result['prediction_id'] = prediction_id
            print(f"Tahmin kaydedildi (ID: {prediction_id})")
        except Exception as e:
            print(f"Tahmin kaydetme hatası: {e}")
            result['prediction_id'] = None
        return result

    def update_result(self, prediction_id, actual_value):
        try:
            success = update_prediction_result(prediction_id, actual_value, self.db_path)
            if success:
                correct_models = []
                # df = load_data_from_sqlite(self.db_path, limit=300) # Her seferinde yüklemek yerine bir önceki predict_next'teki recent_values kullanılabilir mi?
                                                                   # Şimdilik orijinal mantıkta bırakalım

                # Doğru tahmin yapan modelleri belirlemek için bir önceki (tahmin yapılan) diziyi almamız lazım.
                # Bu bilgi şu anki yapıda doğrudan bu metoda gelmiyor.
                # Basitlik adına, şimdilik bu kısmı ensemble'ın kendi içindeki performans takibine bırakabiliriz
                # veya daha karmaşık bir durum yönetimi gerekir.
                # Geçici olarak, sadece ensemble'a genel bir sinyal verelim:

                # Örnek: Tahminin genel olarak başarılı olup olmadığını belirle
                # (Bu kısım sizin "doğru tahmin" tanımınıza göre çok daha detaylı olmalı)
                # Şu anki JetXPredictor.predict_next() metodundan dönen `result` içinde `above_threshold` var.
                # Bunu `predictions` tablosundan çekip `actual_value` ile karşılaştırabiliriz.

                # Ensemble ağırlıklarını güncelle (Bu kısım için ensemble'a doğru/yanlış bilgisi gitmeli)
                # WeightedEnsemble.update_weights metodu şu an 'correct_predictions' listesi bekliyor.
                # Bu listeyi oluşturmak için her bir modelin son yaptığı tahmini (veya olasılığı) bilmemiz
                # ve bunu actual_value ile karşılaştırmamız gerekir. Bu, mevcut yapıda biraz dolaylı.
                # Şimdilik update_weights'i çağırmak için boş bir liste veya daha basit bir mantıkla
                # (örneğin tüm modelleri başarılı/başarısız saymak) ilerleyebiliriz.
                # VEYA WeightedEnsemble'ın update_weights metodunu değiştirebiliriz.
                # ŞİMDİLİK, bu kısmı daha sonra detaylandırmak üzere basitleştirelim:

                if self.ensemble and hasattr(self.ensemble, 'update_weights'):
                     # Gerçek bir "correct_models" listesi oluşturmak için daha fazla mantık gerekir.
                     # Örneğin, en son prediction_id'ye ait 'above_threshold' bilgisini DB'den çekip
                     # actual_value ile karşılaştırarak genel bir "ensemble tahmini doğru muydu?" bilgisi elde edilebilir.
                     # Ve bu bilgi tüm modellere yansıtılabilir veya her modelin son tahminini ayrıca saklamak gerekir.
                     # Basitlik için, tüm modelleri içeren bir liste veya boş liste geçilebilir şimdilik.
                     # Veya WeightedEnsemble.update_weights'i sadece actual_value ile çağıracak şekilde
                     # (veya ensemble tahmininin kendisiyle) güncelleyebilirsiniz.
                     # Sizin kodunuzda bu döngü vardı, onu koruyalım ama verimlilik ve doğruluk açısından gözden geçirilmeli.
                    temp_correct_models = []
                    for name, model in self.models.items():
                        if hasattr(model, 'predict_next_value'):
                            df_hist = load_data_from_sqlite(self.db_path, limit=301) # Tahmin anındaki durumu yakalamak için
                            if df_hist is not None and len(df_hist) > 1: # En az bir önceki değer ve onun öncesi lazım
                                # Son değer actual_value, ondan önceki dizi modelin girdisiydi
                                # Ancak predict_next'e giden recent_values zaten son 300'dü.
                                # Bu mantık biraz karmaşıklaşıyor, çünkü modelin tam olarak hangi girdiyle tahmin yaptığını
                                # burada yeniden oluşturmak zor.
                                # Şimdilik, WeightedEnsemble'ın update_weights metodunun
                                # sadece `correct_models` listesiyle çağrıldığını varsayalım.
                                # Bu listenin nasıl doldurulacağı önemli bir tasarım kararı.
                                # Mevcut kodunuzdaki mantığı koruyarak devam edelim:
                                recent_values_for_eval = df_hist['value'].values[:-1] # Son actual_value hariç
                                if len(recent_values_for_eval) >= max(self.pattern_lengths if self.pattern_lengths else [10]): # En uzun pattern veya min. bir değer
                                    try:
                                        m_pred_val, m_above_prob = model.predict_next_value(recent_values_for_eval)[:2]
                                        is_model_correct = (m_above_prob >= 0.5 and actual_value >= self.threshold) or \
                                                           (m_above_prob < 0.5 and actual_value < self.threshold)
                                        if is_model_correct:
                                            temp_correct_models.append(name)
                                    except:
                                        pass # Model tahmin üretemezse atla
                    self.ensemble.update_weights(temp_correct_models, actual_value, self.threshold) # <<<--- YENİ HALİ
                    correct_models = temp_correct_models # Sadece print için

                # Güven tahmincisini güncelle
                # df = load_data_from_sqlite(self.db_path, limit=1) # Bu en son eklenen değeri alır, tahmin yapılan değeri değil.
                # Bu mantık da gözden geçirilmeli. add_prediction'a neyin prediction, neyin actual gittiği önemli.
                # Şimdilik orijinal kodunuzdaki gibi bırakıyorum:
                df_last_val = load_data_from_sqlite(self.db_path, limit=1) # En son eklenen gerçek değer
                if df_last_val is not None and not df_last_val.empty:
                    # last_prediction_value aslında bir önceki turun tahmini olmalı.
                    # Bu bilgiyi saklamak veya DB'den çekmek gerekir.
                    # Tahminler tablosundan prediction_id'ye ait predicted_value'yu çekelim.
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT predicted_value FROM predictions WHERE id = ?", (prediction_id,))
                    pred_row = cursor.fetchone()
                    conn.close()
                    if pred_row:
                        last_predicted_val_for_this_actual = pred_row[0]
                        if self.confidence_estimator and hasattr(self.confidence_estimator, 'add_prediction'):
                            conn = sqlite3.connect(self.db_path)
                            cursor = conn.cursor()
                            cursor.execute("SELECT confidence_score FROM predictions WHERE id = ?", (prediction_id,))
                            conf_row = cursor.fetchone()
                            conn.close()

                            # --- YENİ KONTROL ---
                            initial_confidence_from_db = conf_row[0] if conf_row and conf_row[0] is not None else 0.5
                            # NaN (Not a Number) gelme ihtimaline karşı da kontrol edelim
                            if np.isnan(initial_confidence_from_db):
                                initial_confidence_to_use = 0.5
                            else:
                                initial_confidence_to_use = initial_confidence_from_db
                            # --- YENİ KONTROL BİTTİ ---

                            self.confidence_estimator.add_prediction(
                                last_predicted_val_for_this_actual, 
                                actual_value, 
                                initial_confidence_to_use # Güncellenmiş değişkeni kullan
                            )

                print(f"Sonuç güncellendi: {prediction_id} -> {actual_value}")
                if 'correct_models' in locals(): # Eğer tanımlandıysa yazdır
                    print(f"Doğru tahmin yapan modeller: {correct_models}")
            return success
        except Exception as e:
            print(f"Sonuç güncelleme hatası: {e}")
            return False

    def add_new_result(self, value):
        try:
            record_id = save_result_to_sqlite(value, self.db_path)
            print(f"Yeni sonuç kaydedildi: {value} (ID: {record_id})")
            return record_id
        except Exception as e:
            print(f"Sonuç kaydetme hatası: {e}")
            return None

    def get_model_info(self):
        if self.ensemble:
            return self.ensemble.get_model_info()
        return {}

    def retrain_models(self):
        df = self.load_data()
        if df is not None and len(df) > 0:
            self.initialize_models(df)
            return True
        return False

# ------------------- STREAMLIT UI KODU (Colab'da doğrudan çalışmaz) -------------------
# Bu kısım Colab'da çalıştırılan ana script içinde kullanılmayacak,
# ancak app.py dosyasının bütünlüğü için burada bırakıyorum.
# Eğer bu dosyayı `streamlit run app.py` ile çalıştırırsanız bu UI devreye girer.

if __name__ == '__main__': # Colab'da import edildiğinde bu blok çalışmaz
    st.set_page_config(
        page_title="JetX Tahmin Uygulaması",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .big-font {font-size:30px !important; font-weight: bold;}
        .result-box {padding: 20px; border-radius: 10px; margin-bottom: 20px;}
        .high-confidence {background-color: rgba(0, 255, 0, 0.2);}
        .medium-confidence {background-color: rgba(255, 255, 0, 0.2);}
        .low-confidence {background-color: rgba(255, 0, 0, 0.2);}
        .above-threshold {color: green; font-weight: bold;}
        .below-threshold {color: red; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

    if 'predictor' not in st.session_state:
        # Streamlit context'inde db_path farklı olabilir, özellikle deployment için.
        # Şimdilik aynı db_path'i kullanıyoruz.
        streamlit_db_path = "jetx_data.db" # Streamlit çalışırken bulunduğu dizine göre

        # Eğer bu dosya bir alt dizindeyse ve db_path göreceli ise, doğru yolu bulmak için:
        # script_dir = os.path.dirname(__file__)
        # streamlit_db_path = os.path.join(script_dir, "jetx_data.db")

        st.session_state.predictor = JetXPredictor(db_path=streamlit_db_path)
        df = st.session_state.predictor.load_data()
        if df is not None and len(df) > 0:
            st.session_state.predictor.initialize_models(df)
        else:
            st.warning("Veritabanında veri bulunamadı. Önce veri ekleyin.")

    if 'last_values' not in st.session_state:
        st.session_state.last_values = []
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'should_retrain' not in st.session_state:
        st.session_state.should_retrain = False
    if 'new_data_count' not in st.session_state:
        st.session_state.new_data_count = 0

    def streamlit_add_value_and_predict(value):
        if 'last_prediction' in st.session_state and st.session_state.last_prediction:
            with st.spinner("Önceki tahmin sonucu güncelleniyor..."):
                success = st.session_state.predictor.update_result(
                    st.session_state.last_prediction['prediction_id'], value
                )
                if success: st.success(f"Önceki tahmin için sonuç güncellendi: {value}")

        record_id = st.session_state.predictor.add_new_result(value)
        if record_id:
            st.session_state.last_values.append(value)
            if len(st.session_state.last_values) > 15:
                st.session_state.last_values = st.session_state.last_values[-15:]
            st.session_state.new_data_count += 1
            if st.session_state.new_data_count >= 10: # Bu değeri de JetXPredictor'dan alabiliriz
                st.session_state.should_retrain = True
                st.session_state.new_data_count = 0
            with st.spinner("Tahmin yapılıyor..."):
                prediction = st.session_state.predictor.predict_next()
                if prediction: st.session_state.last_prediction = prediction
            return True
        else:
            st.error("Değer eklenirken bir hata oluştu.")
            return False

    st.markdown('<p class="big-font">JetX Tahmin Uygulaması</p>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Veri Girişi")
        with st.expander("Tek Değer Girişi", expanded=True):
            with st.form(key="single_value_form"):
                value_input = st.number_input("JetX Değeri Girin:", min_value=1.0, max_value=3000.0, value=1.5, step=0.01, format="%.2f")
                submit_button = st.form_submit_button("Değeri Ekle")
                if submit_button:
                    if streamlit_add_value_and_predict(value_input):
                        st.success(f"Değer eklendi: {value_input}")
                        st.rerun()
        with st.expander("Toplu Değer Girişi"):
            with st.form(key="bulk_value_form"):
                bulk_text = st.text_area("Her satıra bir değer gelecek şekilde değerleri girin:", height=200, help="Örnek:\n1.55\n2.89\n1.56")
                submit_bulk_button = st.form_submit_button("Toplu Değerleri Ekle")
                if submit_bulk_button:
                    lines = bulk_text.strip().split('\n')
                    success_count = 0; error_count = 0
                    progress_bar = st.progress(0)
                    for i, line in enumerate(lines):
                        try:
                            value = float(line.strip())
                            if 1.0 <= value <= 3000.0:
                                if streamlit_add_value_and_predict(value): success_count += 1
                            else: error_count += 1
                        except: error_count += 1
                        progress_bar.progress((i + 1) / len(lines))
                    st.success(f"{success_count} değer başarıyla eklendi. {error_count} değer eklenemedi.")
                    if st.session_state.should_retrain:
                        with st.spinner("Modeller güncelleniyor..."):
                            st.session_state.predictor.retrain_models()
                        st.session_state.should_retrain = False
                    st.rerun()
        st.subheader("Son 15 Veri")
        if st.session_state.last_values:
            df_last = pd.DataFrame({'Sıra': range(1, len(st.session_state.last_values) + 1), 'Değer': st.session_state.last_values})
            st.dataframe(df_last, width=300)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(len(st.session_state.last_values)), st.session_state.last_values, 'bo-')
            ax.axhline(y=1.5, color='r', linestyle='--', alpha=0.7)
            ax.set_xlabel('Sıra'); ax.set_ylabel('Değer'); ax.set_title('Son Değerler')
            st.pyplot(fig)
        else: st.info("Henüz veri girilmedi.")
    with col2:
        st.subheader("Tahmin Yap")
        if st.button("Yeni Tahmin Yap", key="predict_button"):
            if st.session_state.last_values:
                with st.spinner("Tahmin yapılıyor..."):
                    prediction = st.session_state.predictor.predict_next()
                    if prediction: st.session_state.last_prediction = prediction
                    else: st.error("Tahmin yapılamadı. Yeterli veri yok.")
            else: st.warning("Tahmin için önce veri girmelisiniz.")
        if st.session_state.last_prediction:
            prediction = st.session_state.last_prediction
            confidence = prediction.get('confidence_score', 0.0) or 0.0 # None ise 0.0
            predicted_value_display = f"{prediction.get('predicted_value', 0.0):.2f}" if prediction.get('predicted_value') is not None else "N/A"
            above_threshold_prob_display = f"{prediction.get('above_threshold_probability', 0.0):.2f}" if prediction.get('above_threshold_probability') is not None else "N/A"

            confidence_class = "high-confidence" if confidence >= 0.7 else ("medium-confidence" if confidence >= 0.4 else "low-confidence")
            threshold_class = "above-threshold" if prediction.get('above_threshold', False) else "below-threshold"
            threshold_text = "1.5 ÜZERİNDE" if prediction.get('above_threshold', False) else "1.5 ALTINDA"
            st.markdown(f"""
            <div class="result-box {confidence_class}" style="padding: 20px; border: 2px solid {'green' if prediction.get('above_threshold', False) else 'red'}; border-radius: 10px;">
                <h2 style="text-align: center;">Tahmin Sonucu</h2>
                <h3 style="text-align: center; margin-bottom: 20px;" class="{threshold_class}">{threshold_text}</h3>
                <p style="font-size: 18px;">Tahmini değer: <b>{predicted_value_display}</b></p>
                <p style="font-size: 18px;">1.5 üstü olasılığı: <b>{above_threshold_prob_display}</b></p>
                <p style="font-size: 18px;">Güven skoru: <b>{confidence:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
        st.subheader("Model Performansı")
        model_info = st.session_state.predictor.get_model_info()
        if model_info:
            model_data_for_df = []
            for name, info in model_info.items():
                if isinstance(info, dict): # Beklenen format
                    model_data_for_df.append({
                        'Model': name,
                        'Doğruluk': f"{info.get('accuracy', 0)*100:.1f}%",
                        'Ağırlık': f"{info.get('weight', 0):.2f}"
                    })
                else: # Beklenmedik format, logla veya atla
                    print(f"Model info for {name} is not a dict: {info}")

            if model_data_for_df:
                model_df = pd.DataFrame(model_data_for_df)
                st.dataframe(model_df)
                fig, ax = plt.subplots(figsize=(10, 4))
                models = model_df['Model'].tolist()
                accuracies = [float(acc.strip('%'))/100 for acc in model_df['Doğruluk'].tolist()]
                weights = model_df['Ağırlık'].tolist()
                x = np.arange(len(models)) # x ekseni için sayısal değerler
                width = 0.35
                ax.bar(x - width/2, accuracies, width, label='Doğruluk')
                ax.bar(x + width/2, weights, width, label='Ağırlık')
                ax.set_xlabel('Model'); ax.set_ylabel('Değer'); ax.set_title('Model Performansı')
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=45, ha="right")
                ax.legend(); ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
                st.pyplot(fig)
        else: st.info("Henüz model performans bilgisi yok.")
    if st.session_state.should_retrain:
        with st.spinner("Modeller güncelleniyor..."):
            st.session_state.predictor.retrain_models()
        st.session_state.should_retrain = False
    st.sidebar.title("Ayarlar ve Bilgiler")
    st.sidebar.subheader("Veritabanı İstatistikleri")
    df_sidebar = st.session_state.predictor.load_data() # predictor'ı session_state'ten al
    if df_sidebar is not None:
        st.sidebar.info(f"Toplam kayıt sayısı: {len(df_sidebar)}")
        if len(df_sidebar) > 0:
            st.sidebar.info(f"Ortalama değer: {df_sidebar['value'].mean():.2f}")
            st.sidebar.info(f"1.5 üstü oranı: {(df_sidebar['value'] >= 1.5).mean():.2f}")
    else: st.sidebar.warning("Veritabanı verisi yüklenemedi.")
    if st.sidebar.button("Modelleri Yeniden Eğit"):
        with st.spinner("Modeller yeniden eğitiliyor..."):
            success = st.session_state.predictor.retrain_models()
            if success: st.sidebar.success("Modeller başarıyla yeniden eğitildi.")
            else: st.sidebar.error("Modeller eğitilirken bir hata oluştu.")
    st.sidebar.subheader("Hakkında")
    st.sidebar.info("Bu uygulama, JetX oyunu sonuçlarını analiz etmek ve tahminler yapmak için geliştirilmiştir.")
    st.sidebar.warning("Uyarı: Bu uygulama sadece bilimsel amaçlar içindir ve kumar oynamayı teşvik etmez.")
    st.sidebar.text(f"Son güncelleme: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")