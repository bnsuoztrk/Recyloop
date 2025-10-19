"""
predict_price.py
=================
Bi-LSTM + LightGBM Ensemble tahmin aracı
Bir ürünün markası, kategorisi, açıklaması vb. bilgileriyle tahmini fiyat (TL) döndürür.
"""

import json, pickle, argparse, re
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========= Dosya yolları =========
BILSTM_MODEL_PATH = "bilstm_full_model.keras"
TOKENIZER_PATH = "tokenizer.json"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "cat_encoders.pkl"
LGB_MODEL_PATH = "lgb_model.txt"

# ========= Ensemble ağırlıkları =========
WEIGHT_LSTM = 0.7
WEIGHT_LGBM = 0.3

# ========= Yardımcı Fonksiyonlar =========
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9ğüşıöç\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_text_input(data_dict):
    text_parts = []
    for key in ["brand", "description", "subcategory", "main_category"]:
        if key in data_dict and data_dict[key]:
            text_parts.append(str(data_dict[key]))
    return clean_text(" ".join(text_parts))

# ========= Modelleri yükle =========
print("🔹 Modeller yükleniyor...")
bilstm_model = load_model(BILSTM_MODEL_PATH)
lgb_model = lgb.Booster(model_file=LGB_MODEL_PATH)

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)

scaler = pickle.load(open(SCALER_PATH, "rb"))
cat_encoders = pickle.load(open(ENCODERS_PATH, "rb"))

print("✅ Modeller ve yardımcı nesneler yüklendi.")

# ========= Tahmin fonksiyonu =========
def predict_price(brand, main_category, sub_category, description, numeric_features=None):
    data = {
        "brand": brand,
        "main_category": main_category,
        "subcategory": sub_category,
        "description": description,
    }

    # Text girişi
    text_input = build_text_input(data)
    seq = tokenizer.texts_to_sequences([text_input])
    X_pad = pad_sequences(seq, maxlen=bilstm_model.input_shape[0][1], padding="post")

    # Kategorik encoding
    cat_inputs = []
    lgb_input = {}

    for col, encoder in cat_encoders.items():
        if col in data:
            val = str(data[col])
            if val in encoder.classes_:
                encoded_val = encoder.transform([val])[0]
            else:
                encoded_val = 0
        else:
            encoded_val = 0
        cat_inputs.append(np.array([encoded_val]))
        lgb_input[col] = encoded_val

    # Sayısal girdi varsa (örnek: stok, rating vb.)
    if numeric_features is not None:
        num_arr = scaler.transform([numeric_features])
    else:
        num_arr = np.zeros((1, scaler.mean_.shape[0])) if hasattr(scaler, "mean_") else None

    # LSTM tahmini
    lstm_inputs = [X_pad] + cat_inputs + ([num_arr] if num_arr is not None else [])
    pred_lstm = np.expm1(bilstm_model.predict(lstm_inputs, verbose=0))[0][0]

    # LightGBM tahmini
    df_lgb = pd.DataFrame([lgb_input])
    pred_lgb = np.expm1(lgb_model.predict(df_lgb))[0]

    # Ensemble
    price_pred = WEIGHT_LSTM * pred_lstm + WEIGHT_LGBM * pred_lgb
    return round(price_pred, 2)

# ========= CLI arayüzü =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ürün bilgisiyle fiyat tahmini yapar.")
    parser.add_argument("--brand", type=str, required=True, help="Ürün markası")
    parser.add_argument("--main_category", type=str, required=True, help="Ana kategori (örnek: Giyim, Canta, Ayakkabi)")
    parser.add_argument("--subcategory", type=str, required=True, help="Alt kategori (örnek: Elbise, Babet, Sirt_Cantasi)")
    parser.add_argument("--description", type=str, required=True, help="Ürün açıklaması")

    args = parser.parse_args()
    pred = predict_price(
        brand=args.brand,
        main_category=args.main_category,
        sub_category=args.subcategory,
        description=args.description
    )
    print(f"\n💰 Tahmini fiyat: {pred} TL")
