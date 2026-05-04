"""
╔══════════════════════════════════════════════════════════════╗
║    Bengaluru Power Load Forecasting — Hybrid XGBoost + LSTM  ║
║    Final Year Project | One-Shot Run                         ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python bengaluru_load_forecast.py
    → When prompted, enter the folder containing your CSV file(s)
      OR press Enter to auto-detect from the current directory.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for all machines)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
TIME_STEPS   = 24        # LSTM look-back window (hours)
XGB_WEIGHT   = 0.5       # Weight for XGBoost in hybrid blend
LSTM_WEIGHT  = 0.5       # Weight for LSTM in hybrid blend
TRAIN_RATIO  = 0.80      # 80% train / 20% test
LSTM_EPOCHS  = 15        # Increase to 30+ for better accuracy (slower)
LSTM_UNITS   = 64
OUTPUT_DIR   = "results" # Folder where plots + CSV are saved

# Features to use — all columns present in your Bengaluru dataset
FEATURE_COLS = [
    'hour', 'day_of_week', 'month',
    'is_weekend', 'is_holiday', 'is_festival',
    'is_pre_holiday', 'is_post_holiday',
    'is_pre_festival', 'is_post_festival',
    'workday_after_holiday',
    'temperature', 'humidity', 'wind_speed', 'solar_irradiance',
    'lag_1', 'lag_24', 'lag_168',
    'rolling_mean_24', 'rolling_std_24',
]
TARGET_COL = 'load'

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)

def metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    print(f"  {name:<12}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

# ──────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────
section("1. LOADING DATA")

folder = input("  Enter folder path containing CSV file(s) [press Enter for current dir]: ").strip()
if not folder:
    folder = "."

files = sorted(glob.glob(os.path.join(folder, "*.csv")))
if not files:
    print(f"  ❌ No CSV files found in '{folder}'. Exiting.")
    exit(1)

print(f"  Found {len(files)} file(s):")
for f in files:
    print(f"    • {os.path.basename(f)}")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"  ✅ Loaded {len(df):,} rows, {df.shape[1]} columns")

# ──────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────────────────────
section("2. PREPROCESSING")

# Parse datetime
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
before = len(df)
df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
print(f"  Rows after datetime parse: {len(df):,} (dropped {before - len(df)} invalid)")

# Keep only needed columns
available_features = [c for c in FEATURE_COLS if c in df.columns]
missing_features   = [c for c in FEATURE_COLS if c not in df.columns]

if missing_features:
    print(f"  ⚠  Missing columns (will be skipped): {missing_features}")

print(f"  Using {len(available_features)} feature columns + target '{TARGET_COL}'")

df_model = df[['datetime'] + available_features + [TARGET_COL]].dropna().reset_index(drop=True)
print(f"  Final dataset: {len(df_model):,} rows  ({df_model['datetime'].min()} → {df_model['datetime'].max()})")

X = df_model[available_features]
y = df_model[TARGET_COL]

# Train / test split (chronological — never random for time-series!)
split = int(len(df_model) * TRAIN_RATIO)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

# ──────────────────────────────────────────────────────────────
# 3. XGBOOST
# ──────────────────────────────────────────────────────────────
section("3. XGBOOST MODEL")

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_pred_xgb = xgb.predict(X_test)
print("  ✅ XGBoost trained")

# ──────────────────────────────────────────────────────────────
# 4. LSTM
# ──────────────────────────────────────────────────────────────
section("4. LSTM MODEL")

# Scale features for LSTM
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

def make_sequences(X_arr, y_arr, t=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X_arr) - t):
        Xs.append(X_arr[i:i+t])
        ys.append(y_arr[i+t])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_sequences(X_scaled, y_scaled)

# Align split with sequence offset
split_seq = split - TIME_STEPS
if split_seq < 0:
    split_seq = int(len(X_seq) * TRAIN_RATIO)

X_tr_seq, X_te_seq = X_seq[:split_seq], X_seq[split_seq:]
y_tr_seq, y_te_seq = y_seq[:split_seq], y_seq[split_seq:]

lstm = Sequential([
    LSTM(LSTM_UNITS, return_sequences=True,
         input_shape=(TIME_STEPS, len(available_features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.1),
    Dense(1),
])
lstm.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
history = lstm.fit(
    X_tr_seq, y_tr_seq,
    epochs=LSTM_EPOCHS,
    batch_size=64,
    validation_split=0.1,
    callbacks=[es],
    verbose=0,
)
print(f"  ✅ LSTM trained ({len(history.history['loss'])} epochs)")

# Inverse-transform LSTM predictions back to MW
y_pred_lstm_scaled = lstm.predict(X_te_seq, verbose=0).flatten()
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1,1)).flatten()

# Align all arrays to the same length
min_len = min(len(y_pred_xgb), len(y_pred_lstm))
y_pred_xgb  = y_pred_xgb[-min_len:]
y_pred_lstm = y_pred_lstm[-min_len:]
y_true      = y_test.values[-min_len:]

# ──────────────────────────────────────────────────────────────
# 5. HYBRID MODEL
# ──────────────────────────────────────────────────────────────
section("5. HYBRID MODEL")

# Optional: tune weights by minimising RMSE on a validation slice
val_len   = min_len // 5
rmse_xgb  = np.sqrt(mean_squared_error(y_true[:val_len], y_pred_xgb[:val_len]))
rmse_lstm = np.sqrt(mean_squared_error(y_true[:val_len], y_pred_lstm[:val_len]))

# Inverse-RMSE weighting (better model gets higher weight)
w_xgb  = (1/rmse_xgb)  / (1/rmse_xgb + 1/rmse_lstm)
w_lstm = (1/rmse_lstm) / (1/rmse_xgb + 1/rmse_lstm)

y_pred_hybrid = w_xgb * y_pred_xgb + w_lstm * y_pred_lstm

print(f"  Auto-tuned weights → XGBoost: {w_xgb:.2f}  |  LSTM: {w_lstm:.2f}")

# ──────────────────────────────────────────────────────────────
# 6. EVALUATION
# ──────────────────────────────────────────────────────────────
section("6. EVALUATION RESULTS")

results = []
results.append(metrics("XGBoost",  y_true, y_pred_xgb))
results.append(metrics("LSTM",     y_true, y_pred_lstm))
results.append(metrics("Hybrid ★", y_true, y_pred_hybrid))

results_df = pd.DataFrame(results)

# ──────────────────────────────────────────────────────────────
# 7. NEXT-HOUR FORECAST
# ──────────────────────────────────────────────────────────────
section("7. NEXT-HOUR FORECAST")

last_row = X.iloc[[-1]].copy()
next_hour_pred = float(xgb.predict(last_row)[0])
reserve_needed = next_hour_pred * 0.20

print(f"  🔮 Next hour predicted load : {next_hour_pred:,.1f} MW")
print(f"  ⚡ Spinning reserve (20%)   : {reserve_needed:,.1f} MW")

# Anomaly detection (>2σ above mean)
threshold = np.mean(y_true) + 2 * np.std(y_true)
if next_hour_pred > threshold:
    print(f"  ⚠️  ANOMALY WARNING — Prediction {next_hour_pred:,.1f} MW exceeds threshold {threshold:,.1f} MW")
else:
    print(f"  ✅ Load within normal range (threshold: {threshold:,.1f} MW)")

# ──────────────────────────────────────────────────────────────
# 8. SCENARIO TESTING
# ──────────────────────────────────────────────────────────────
section("8. SCENARIO ANALYSIS")

scenarios = {
    "Normal":          last_row.copy(),
    "Heatwave (+5°C)": last_row.copy(),
    "Festival Load":   last_row.copy(),
    "Rainy Day":       last_row.copy(),
}
if 'temperature'     in scenarios["Heatwave (+5°C)"].columns:
    scenarios["Heatwave (+5°C)"]['temperature']   += 5
if 'is_festival'     in scenarios["Festival Load"].columns:
    scenarios["Festival Load"]['is_festival']      = 1
if 'humidity'        in scenarios["Rainy Day"].columns:
    scenarios["Rainy Day"]['humidity']             += 20
if 'solar_irradiance' in scenarios["Rainy Day"].columns:
    scenarios["Rainy Day"]['solar_irradiance']     *= 0.3

print(f"  {'Scenario':<22}  Predicted Load (MW)  Δ vs Normal")
base = None
for name, row in scenarios.items():
    pred = float(xgb.predict(row)[0])
    if base is None:
        base = pred
    delta = pred - base
    sign  = "+" if delta >= 0 else ""
    print(f"  {name:<22}  {pred:>10,.1f} MW       {sign}{delta:,.1f} MW")

# ──────────────────────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────
section("9. TOP FEATURE IMPORTANCE (XGBoost)")

importance = pd.Series(xgb.feature_importances_, index=available_features)
top10 = importance.nlargest(10)
for feat, score in top10.items():
    bar = "█" * int(score * 200)
    print(f"  {feat:<22} {bar} {score:.4f}")

# ──────────────────────────────────────────────────────────────
# 10. PLOTS (saved to results/)
# ──────────────────────────────────────────────────────────────
section("10. GENERATING PLOTS")

test_dates = df_model['datetime'].iloc[-min_len:].reset_index(drop=True)
plot_n     = min(min_len, 7 * 24)   # show up to 7 days

# ── Fig 1: Main prediction plot ──
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
fig.suptitle("Bengaluru Power Load Forecasting — Hybrid XGBoost + LSTM", fontsize=14, weight='bold')

ax = axes[0]
ax.plot(test_dates[:plot_n], y_true[:plot_n],       lw=1.5, label="Actual",   color='#1f77b4')
ax.plot(test_dates[:plot_n], y_pred_hybrid[:plot_n],lw=1.5, label="Hybrid ★", color='#d62728', alpha=0.85)
ax.set_title("Actual vs Hybrid Prediction (first 7 days of test set)")
ax.set_ylabel("Load (MW)"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(test_dates[:plot_n], y_true[:plot_n],       lw=1,   label="Actual",   color='#1f77b4')
ax.plot(test_dates[:plot_n], y_pred_xgb[:plot_n],  lw=1,   label="XGBoost",  color='#ff7f0e', alpha=0.8)
ax.plot(test_dates[:plot_n], y_pred_lstm[:plot_n], lw=1,   label="LSTM",     color='#2ca02c', alpha=0.8)
ax.set_title("Individual Models vs Actual")
ax.set_ylabel("Load (MW)"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
residuals = y_true[:plot_n] - y_pred_hybrid[:plot_n]
ax.bar(range(len(residuals)), residuals, color=np.where(residuals > 0, '#2ca02c', '#d62728'), alpha=0.7)
ax.axhline(0, color='black', lw=0.8)
ax.set_title("Prediction Residuals (Actual − Hybrid)")
ax.set_ylabel("Residual (MW)"); ax.set_xlabel("Hour index"); ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/prediction_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: {OUTPUT_DIR}/prediction_plots.png")

# ── Fig 2: Metrics comparison ──
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Model Performance Comparison", fontsize=13, weight='bold')

models    = [r['model'] for r in results]
colors    = ['#ff7f0e', '#2ca02c', '#d62728']
metrics_k = [('RMSE', 'RMSE (MW)'), ('MAE', 'MAE (MW)'), ('MAPE', 'MAPE (%)')]

for ax, (key, label) in zip(axes, metrics_k):
    vals = [r[key] for r in results]
    bars = ax.bar(models, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(label); ax.set_ylabel(label); ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f"{v:.2f}", ha='center', va='bottom', fontsize=9, weight='bold')

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: {OUTPUT_DIR}/model_comparison.png")

# ── Fig 3: Feature importance ──
fig, ax = plt.subplots(figsize=(10, 6))
top10_sorted = top10.sort_values()
colors_bar   = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10_sorted)))
top10_sorted.plot(kind='barh', ax=ax, color=colors_bar)
ax.set_title("Top 10 Feature Importance (XGBoost)", fontsize=13, weight='bold')
ax.set_xlabel("Importance Score")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: {OUTPUT_DIR}/feature_importance.png")

# ── Fig 4: LSTM training loss ──
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history['loss'],     label='Train Loss', color='#1f77b4')
ax.plot(history.history['val_loss'], label='Val Loss',   color='#d62728')
ax.set_title("LSTM Training Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/lstm_training_loss.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: {OUTPUT_DIR}/lstm_training_loss.png")

# ── Fig 5: 24-hour load profile ──
hourly_avg = pd.DataFrame({'hour': df_model['hour'], 'load': df_model[TARGET_COL]})
profile    = hourly_avg.groupby('hour')['load'].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(profile.index, profile.values, alpha=0.3, color='#1f77b4')
ax.plot(profile.index, profile.values, marker='o', ms=5, lw=2, color='#1f77b4')
ax.set_title("Average 24-Hour Load Profile — Bengaluru", fontsize=13, weight='bold')
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Average Load (MW)")
ax.set_xticks(range(0, 24, 2)); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/daily_load_profile.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: {OUTPUT_DIR}/daily_load_profile.png")

# ──────────────────────────────────────────────────────────────
# 11. SAVE RESULTS CSV
# ──────────────────────────────────────────────────────────────
section("11. SAVING RESULTS")

out_df = pd.DataFrame({
    'datetime':        test_dates.values,
    'actual_load':     y_true,
    'xgb_prediction':  y_pred_xgb,
    'lstm_prediction': y_pred_lstm,
    'hybrid_prediction': y_pred_hybrid,
    'error_mw':        y_true - y_pred_hybrid,
})
out_path = f"{OUTPUT_DIR}/predictions.csv"
out_df.to_csv(out_path, index=False)
print(f"  ✅ Saved: {out_path}")

results_df.to_csv(f"{OUTPUT_DIR}/model_metrics.csv", index=False)
print(f"  ✅ Saved: {OUTPUT_DIR}/model_metrics.csv")

# ──────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────
section("✅ COMPLETE — SUMMARY")

best = results_df.loc[results_df['RMSE'].idxmin()]
print(f"""
  Best model   : {best['model']}
  RMSE         : {best['RMSE']:.2f} MW
  MAE          : {best['MAE']:.2f} MW
  R²           : {best['R2']:.4f}
  MAPE         : {best['MAPE']:.2f}%

  Next hour    : {next_hour_pred:,.1f} MW
  Reserve (20%): {reserve_needed:,.1f} MW

  All plots saved to → ./{OUTPUT_DIR}/
  ├── prediction_plots.png
  ├── model_comparison.png
  ├── feature_importance.png
  ├── lstm_training_loss.png
  └── daily_load_profile.png
""")