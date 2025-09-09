from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import joblib

df = pd.read_csv("Limao.csv")
# Convert 'Tanggal' to datetime objects
df['Tanggal'] = pd.to_datetime(df['Tanggal'])

for i in range(1, 3):  # t-1, t-2
    df[f'Produksi_Padi_Ton_clean_lag_{i}'] = df.groupby('Kabupaten_Kota')['Produksi_Padi_Ton_clean'].shift(i)

# Create lagged climate variables
for col in ['Suhu_Rata_C_clean', 'Curah_Hujan_mm_clean', 'Kelembapan_Persen_clean']:
    for i in range(1, 3): # t-1, t-2
        df[f'{col}_lag_{i}'] = df.groupby('Kabupaten_Kota')[col].shift(i)

# Handle spatial lag (requires a contiguity matrix, which is not available in the notebook)
# For demonstration, we will skip the spatial lag calculation. If you have a contiguity
# matrix or geospatial data, this step would be performed here.

# Drop rows with NaN values created by lags
df.dropna(inplace=True)

# Define features (X) and target (y)
# Exclude the original 'Produksi_Padi_Ton_clean' and spatial features if not included
features = [col for col in df.columns if col not in ['Tanggal', 'Tahun', 'Bulan', 'Kabupaten_Kota', 'Latitude_dd', 'Longitude_dd', 'Produksi_Padi_Ton_clean']]
X = df[features]
y = df['Produksi_Padi_Ton_clean']

# Normalize/scale all features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


print("Prepared data with lag features and scaled features:")
print(X_scaled.head())
print(y.head())

# Define the window size for temporal lags
lag_window = 6

# Create temporal lag features for 'Produksi_Padi_Ton_clean'
for i in range(1, lag_window + 1):
    df[f'Produksi_Padi_Ton_clean_lag_{i}'] = df.groupby('Kabupaten_Kota')['Produksi_Padi_Ton_clean'].shift(i)

# Create temporal lag features for climate variables
for col in ['Suhu_Rata_C_clean', 'Curah_Hujan_mm_clean', 'Kelembapan_Persen_clean']:
    for i in range(1, lag_window + 1):
        df[f'{col}_lag_{i}'] = df.groupby('Kabupaten_Kota')[col].shift(i)

# Create seasonal features using sine and cosine transformations of the month
df['Month'] = df['Tanggal'].dt.month
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Drop the original 'Month' column as it's no longer needed
df = df.drop('Month', axis=1)

# Drop rows with NaN values created by lags
df.dropna(inplace=True)

# Define features (X) and target (y)
non_feature_cols = ['Tanggal', 'Tahun', 'Bulan', 'Kabupaten_Kota', 'Latitude_dd', 'Longitude_dd', 'Produksi_Padi_Ton_clean']
features = [col for col in df.columns if col not in non_feature_cols]

X = df[features]
y = df['Produksi_Padi_Ton_clean']

# Normalize/scale all features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


print("Prepared data with increased lag features, seasonal features, and scaled features:")
print(X_scaled.head())
print(y.head())
print(X_scaled.shape)
print(y.shape)

# Convert 'Tanggal' to datetime objects
df['Tanggal'] = pd.to_datetime(df['Tanggal'])

print("IncreasingLag,Seosonal, and Scaled features")

# Define the window size for temporal lags
# Increased lag window as requested
lag_window = 12 # Including t-3, t-6, t-12

# Create temporal lag features for 'Produksi_Padi_Ton_clean'
for i in range(1, lag_window + 1):
    df[f'Produksi_Padi_Ton_clean_lag_{i}'] = df.groupby('Kabupaten_Kota')['Produksi_Padi_Ton_clean'].shift(i)

# Create temporal lag features for climate variables
for col in ['Suhu_Rata_C_clean', 'Curah_Hujan_mm_clean', 'Kelembapan_Persen_clean']:
    # Adding lags up to lag_window, which includes t-3, t-6, t-12
    for i in range(1, lag_window + 1):
        df[f'{col}_lag_{i}'] = df.groupby('Kabupaten_Kota')[col].shift(i)

# Create seasonal features using sine and cosine transformations of the month
df['Month'] = df['Tanggal'].dt.month
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Drop the original 'Month' column as it's no longer needed
df = df.drop('Month', axis=1)

# Drop rows with NaN values created by lags
df.dropna(inplace=True)

# Define features (X) and target (y)
non_feature_cols = ['Tanggal', 'Tahun', 'Bulan', 'Kabupaten_Kota', 'Latitude_dd', 'Longitude_dd', 'Produksi_Padi_Ton_clean']
features = [col for col in df.columns if col not in non_feature_cols]

X = df[features]
y = df['Produksi_Padi_Ton_clean']

# Normalize/scale all features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


print("Prepared data with increased lag features, seasonal features, and scaled features:")
print(X_scaled.head())
print(y.head())
print(X_scaled.shape)
print(y.shape)


# Split data into training (2018-2022) and testing (2023-2024) sets based on the original df
# Ensure to use the indices from the split on the original df to split the scaled X and the target y

train_indices = df[(df['Tanggal'].dt.year >= 2018) & (df['Tanggal'].dt.year <= 2022)].index
test_indices = df[(df['Tanggal'].dt.year >= 2023) & (df['Tanggal'].dt.year <= 2024)].index

X_train_scaled = X_scaled.loc[train_indices]
y_train = y.loc[train_indices]

X_test_scaled = X_scaled.loc[test_indices]
y_test = y.loc[test_indices]


print("Data split into training (2018-2022) and testing (2023-2024) sets.")
print("\nTraining set shape (X_train_scaled, y_train):")
print(X_train_scaled.shape, y_train.shape)
print("\nTesting set shape (X_test_scaled, y_test):")
print(X_test_scaled.shape, y_test.shape)
#SMAPE
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Handle the case where both y_true and y_pred are zero to avoid division by zero
    # In this case, the error is 0.
    return np.mean(numerator / np.where(denominator == 0, 1, denominator)) * 100

joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "features.pkl")