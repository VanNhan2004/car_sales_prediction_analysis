import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import learning_curve
# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("dataset/data_new.csv")

feature_cols = [
    "Price",
    "Transmission",
    "Power",
    "Engine CC",
    "Manufacturer_le",
    "Model_freq",
    "Fuel_automatic",
    "Fuel_diesel",
    "Fuel_petrol"
]

X = df[feature_cols]
y = df["Total"]

# =========================
# 2. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# 3. MODEL GỐC (BASELINE)
# =========================
rf_base = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

r2_base = r2_score(y_test, y_pred_base)
mae_base = mean_absolute_error(y_test, y_pred_base)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))


# =========================
# 4. RANDOMIZED SEARCH CV
# =========================
param_dist = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# =========================
# 5. ĐÁNH GIÁ MODEL SAU TỐI ƯU
# =========================
y_pred_best = best_model.predict(X_test)

r2_best = r2_score(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))


# =========================
# 6. SO SÁNH TRƯỚC / SAU
# =========================
print("\n===== COMPARISON =====")
print(f"R2   : {r2_base:.4f}  →  {r2_best:.4f}")
print(f"MAE  : {mae_base:.2f} → {mae_best:.2f}")
print(f"RMSE : {rmse_base:.2f} → {rmse_best:.2f}")

train_sizes, train_scores, test_scores = learning_curve(
    best_model,
    X, y,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)

print("Train scores mean:", train_scores.mean(axis=1))
print("Test scores mean :", test_scores.mean(axis=1))
# =========================
# 7. LƯU MODEL
# =========================
joblib.dump(best_model, "models/best_random_forest.pkl")
print("\n✅ Best model saved to models/best_random_forest.pkl")

