from sklearn.ensemble import RandomForestRegressor
from Data import X_scaled, X_train_scaled, X_test_scaled, y_train, y_test, smape
import joblib

# Instantiate the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the scaled training data
rf_model.fit(X_train_scaled, y_train)

print("Random Forest model trained successfully on scaled data.")

from sklearn.metrics import mean_squared_error
import numpy as np

# Make predictions on the scaled test data
y_pred_rf = rf_model.predict(X_test_scaled)

# Calculate RMSE
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Calculate SMAPE
smape_rf = smape(y_test, y_pred_rf)

# Print the evaluation metrics
print(f"Random Forest Model Evaluation (with increased lags and seasonal features):")
print(f"RMSE: {rmse_rf:.2f}")
print(f"SMAPE: {smape_rf:.2f}%")

joblib.dump(rf_model, "RFM.pkl")