import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from Data import X_scaled, X_train_scaled, X_test_scaled, y_train, y_test, smape
import numpy as np
import joblib

# Instantiate the LightGBM Regressor model
lgbm_model = lgb.LGBMRegressor(random_state=42)

# Fit the instantiated model to the scaled training data
lgbm_model.fit(X_train_scaled, y_train)

print("LightGBM model trained successfully on scaled data.")

# Make predictions on the scaled test data using the trained LightGBM model
y_pred_lgbm = lgbm_model.predict(X_test_scaled)

# Calculate Root Mean Squared Error (RMSE)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))

# Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
smape_lgbm = smape(y_test, y_pred_lgbm)

# Print the calculated RMSE and SMAPE values for the LightGBM model
print("LightGBM Model Evaluation (with increased lags and seasonal features):")
print(f"RMSE: {rmse_lgbm:.2f}")
print(f"SMAPE: {smape_lgbm:.2f}%")


joblib.dump(lgbm_model, "LGBMM.pkl")