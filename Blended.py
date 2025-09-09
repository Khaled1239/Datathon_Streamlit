from RF import y_pred_rf
from LGBM import y_pred_lgbm
import numpy as np
from sklearn.metrics import mean_squared_error
from Data import X_scaled, X_train_scaled, X_test_scaled, y_train, y_test, smape
import pandas as pd

# Note: Ensure y_pred_rf and y_pred_lgbm are available from previous steps

# Implement a simple blending approach by averaging the predictions
# Ensure the predictions are aligned (which they should be if made on the same test set X_test_scaled)
y_pred_blended = (y_pred_rf + y_pred_lgbm) / 2

# Calculate RMSE for the blended model
rmse_blended = np.sqrt(mean_squared_error(y_test, y_pred_blended))

# Calculate SMAPE for the blended model
smape_blended = smape(y_test, y_pred_blended)

# Print the evaluation metrics for the blended model
print("Blended Model Evaluation (Random Forest + LightGBM Averaging):")
print(f"RMSE: {rmse_blended:.2f}")
print(f"SMAPE: {smape_blended:.2f}%")