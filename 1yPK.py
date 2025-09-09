import joblib
import pandas as pd
import numpy as np
from Data import df   # dataset asli dari Data.py
import matplotlib.pyplot as plt

# ================================
# 1. Load scaler, features, dan model
# ================================
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
rf_model = joblib.load("RFM.pkl")
lgbm_model = joblib.load("LGBMM.pkl")

# ================================
# 2. Tentukan periode prediksi
# ================================
last_date = df["Tanggal"].max()
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(),
                             periods=12, freq="MS")

# ================================
# 3. Prediksi tiap Kabupaten
# ================================
predictions_kabupaten = []

for kab in df["Kabupaten_Kota"].unique():
    current_df = df[df["Kabupaten_Kota"] == kab].copy()

    for date in future_dates:
        # Tambahkan row kosong untuk bulan prediksi
        new_row = {col: np.nan for col in current_df.columns}
        new_row["Tanggal"] = date
        new_row["Kabupaten_Kota"] = kab
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

        # Re-generate lag features
        lag_window = 12
        for i in range(1, lag_window + 1):
            current_df[f'Produksi_Padi_Ton_clean_lag_{i}'] = (
                current_df.groupby("Kabupaten_Kota")['Produksi_Padi_Ton_clean'].shift(i)
            )
            for col in ['Suhu_Rata_C_clean', 'Curah_Hujan_mm_clean', 'Kelembapan_Persen_clean']:
                current_df[f'{col}_lag_{i}'] = (
                    current_df.groupby("Kabupaten_Kota")[col].shift(i)
                )

        # Seasonal features
        current_df['Month'] = current_df['Tanggal'].dt.month
        current_df['Month_sin'] = np.sin(2 * np.pi * current_df['Month'] / 12)
        current_df['Month_cos'] = np.cos(2 * np.pi * current_df['Month'] / 12)
        current_df = current_df.drop('Month', axis=1)

        # Ambil fitur terakhir untuk bulan prediksi
        X_latest = current_df[features].iloc[-1:].copy()
        X_latest_scaled = scaler.transform(X_latest)

        # Prediksi
        y_rf = rf_model.predict(X_latest_scaled)[0]
        y_lgbm = lgbm_model.predict(X_latest_scaled)[0]
        y_blend = (y_rf + y_lgbm) / 2

        y_rf = max(0, y_rf)
        y_lgbm = max(0, y_lgbm)
        y_blend = max(0, y_blend)

        # Simpan hasil prediksi
        predictions_kabupaten.append({
            "Kabupaten_Kota": kab,
            "Tanggal": date,
            "RF_Pred": y_rf,
            "LGBM_Pred": y_lgbm,
            "Blended_Pred": y_blend
        })

        # Update kolom target dengan hasil blended
        current_df.loc[current_df.index[-1], "Produksi_Padi_Ton_clean"] = y_blend

# DataFrame hasil prediksi per kabupaten
pred_kabupaten_df = pd.DataFrame(predictions_kabupaten)
pred_kabupaten_df.to_csv("1YPrediction_K.csv", index=False)
print("Saved 1YPrediction_K.csv")