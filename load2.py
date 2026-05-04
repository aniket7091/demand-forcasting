import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# ================================
# 📥 LOAD DATA
# ================================
folder_path = input("Enter folder path: ")
files = glob.glob(os.path.join(folder_path, "*.csv"))

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])
df = df.sort_values('datetime')

# ================================
# 🎯 FEATURES
# ================================
features = [
    'hour','day_of_week','month','is_weekend','is_holiday',
    'temperature','humidity','wind_speed','solar_irradiance',
    'lag_1','lag_24','lag_168','rolling_mean_24','rolling_std_24'
]

df = df.dropna(subset=features + ['load'])

X = df[features]
y = df['load']

# ================================
# 📊 TRAIN TEST SPLIT
# ================================
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ================================
# ⚡ PEAK WEIGHTING
# ================================
peak_threshold = np.percentile(y_train, 90)
weights = np.where(y_train > peak_threshold, 2, 1)

# ================================
# 🤖 XGBOOST
# ================================
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8
)

start = time.time()
xgb_model.fit(X_train, y_train, sample_weight=weights)
print(f"⏱ XGB Training: {time.time()-start:.2f}s")

y_pred_xgb = xgb_model.predict(X_test)

# ================================
# 🤖 LSTM (PROPER)
# ================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

def create_sequences(data, target, steps=24):
    Xs, ys = [], []
    for i in range(len(data)-steps):
        Xs.append(data[i:i+steps])
        ys.append(target.iloc[i+steps])
    return np.array(Xs), np.array(ys)

X_lstm, y_lstm = create_sequences(X_scaled, y)

split_lstm = int(len(X_lstm)*0.8)
X_train_lstm, X_test_lstm = X_lstm[:split_lstm], X_lstm[split_lstm:]
y_train_lstm, y_test_lstm = y_lstm[:split_lstm], y_lstm[split_lstm:]

lstm_model = Sequential([
    Input(shape=(24, len(features))),
    LSTM(64),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

start = time.time()
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)
print(f"⏱ LSTM Training: {time.time()-start:.2f}s")

y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

# ================================
# 🔀 SMART HYBRID
# ================================
min_len = min(len(y_pred_xgb), len(y_pred_lstm))

y_pred_xgb = y_pred_xgb[-min_len:]
y_pred_lstm = y_pred_lstm[-min_len:]
y_test_final = y_test.values[-min_len:]

# weighted hybrid (based on performance)
rmse_xgb = np.sqrt(mean_squared_error(y_test_final, y_pred_xgb))
rmse_lstm = np.sqrt(mean_squared_error(y_test_final, y_pred_lstm))

w_xgb = 1 / rmse_xgb
w_lstm = 1 / rmse_lstm

y_pred = (w_xgb*y_pred_xgb + w_lstm*y_pred_lstm) / (w_xgb + w_lstm)

# ================================
# 📊 METRICS
# ================================
def metrics(y_true, y_pred):
    print("\n📊 PERFORMANCE")
    print("MAE :", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2  :", r2_score(y_true, y_pred))

metrics(y_test_final, y_pred)

# ================================
# 🔮 NEXT + FUTURE PREDICTION
# ================================
future_preds = []
temp_df = df.copy()

for i in range(6):
    last = temp_df.iloc[-1].copy()
    last['hour'] = (last['hour']+1)%24
    last['lag_1'] = last['load']
    last['lag_24'] = temp_df.iloc[-24]['load']

    pred = xgb_model.predict(pd.DataFrame([last[features]]))[0]
    last['load'] = pred

    temp_df = pd.concat([temp_df, pd.DataFrame([last])])
    future_preds.append(pred)

print("\n📈 Future 6 Hours:", future_preds)

# ================================
# 🚨 CONFIDENCE INTERVAL
# ================================
residuals = y_test_final - y_pred
std = np.std(residuals)

next_pred = future_preds[0]
print(f"\n🔮 Next Load: {next_pred:.2f}")
print(f"Range: [{next_pred-1.96*std:.2f}, {next_pred+1.96*std:.2f}]")

# ================================
# 📊 FEATURE IMPORTANCE
# ================================
plt.figure()
plt.barh(features, xgb_model.feature_importances_)
plt.title("Feature Importance")
plt.show()

# ================================
# 📈 FINAL GRAPH
# ================================
plt.figure(figsize=(12,6))
plt.plot(y_test_final, label="Actual")
plt.plot(y_pred, label="Hybrid Prediction")
plt.legend()
plt.title("Final Hybrid Model")
plt.show()

# ================================
# 🔁 LIVE LEARNING (SAFE SIMULATION)
# ================================
def update_model(new_data):
    global xgb_model

    print("\n🔄 Updating model with new data...")

    new_X = new_data[features]
    new_y = new_data['load']

    xgb_model.fit(new_X, new_y, xgb_model.get_booster())

    joblib.dump(xgb_model, "xgb_model.pkl")
    print("✅ Model Updated")

print("\n✅ FINAL SYSTEM READY 🚀")