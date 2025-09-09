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
# 2. Tentukan periode prediksi (12 bulan ke depan)
# ================================
last_date = df["Tanggal"].max()
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(),
                             periods=12, freq="MS")

# ================================
# 3. Prediksi per Kabupaten/Kota
# ================================
all_predictions = []

for daerah in df["Kabupaten_Kota"].unique():
    current_df = df[df["Kabupaten_Kota"] == daerah].copy()
    pred_rf, pred_lgbm, pred_blend = [], [], []

    for date in future_dates:
        # Tambahkan row kosong untuk bulan prediksi
        new_row = {col: np.nan for col in current_df.columns}
        new_row["Tanggal"] = date
        new_row["Kabupaten_Kota"] = daerah
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

        pred_rf.append(y_rf)
        pred_lgbm.append(y_lgbm)
        pred_blend.append(y_blend)

        # Masukkan hasil prediksi ke kolom target (pakai blended untuk update lag)
        current_df.loc[current_df.index[-1], "Produksi_Padi_Ton_clean"] = y_blend

    daerah_pred = pd.DataFrame({
        "Tanggal": future_dates,
        "Kabupaten_Kota": daerah,
        "RF_Pred": pred_rf,
        "LGBM_Pred": pred_lgbm,
        "Blended_Pred": pred_blend
    })
    all_predictions.append(daerah_pred)

# ================================
# 4. Gabungkan semua kabupaten
# ================================
all_predictions = pd.concat(all_predictions)

# Agregasi → total provinsi per bulan
prov_predictions = all_predictions.groupby("Tanggal")[["RF_Pred", "LGBM_Pred", "Blended_Pred"]].sum().reset_index()

print(prov_predictions)

# ================================
# 5. Visualisasi
# ================================
plt.figure(figsize=(12,6))
plt.plot(prov_predictions["Tanggal"], prov_predictions["RF_Pred"], marker='o', label="Random Forest")
plt.plot(prov_predictions["Tanggal"], prov_predictions["LGBM_Pred"], marker='s', label="LightGBM")
plt.plot(prov_predictions["Tanggal"], prov_predictions["Blended_Pred"], marker='^', linewidth=2, label="Blended", color="black")

plt.title("Prediksi Total Produksi Padi Jawa Timur Tahun 2025")
plt.xlabel("Tanggal")
plt.ylabel("Produksi Padi (Ton)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================================
# 6. Simpan ke CSV
# ================================
prov_predictions.to_csv("1YPrediction.csv", index=False)
print("Saved EastJava 1YPrediction.csv")
