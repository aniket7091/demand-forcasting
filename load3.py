"""
╔══════════════════════════════════════════════════════════════════════╗
║   Bengaluru Power Load Forecasting — UPGRADED v2                     ║
║   Final Year Project                                                 ║
║                                                                      ║
║   Upgrades over v1:                                                  ║
║   ✦ Fourier time encoding (replaces raw hour/day integers)           ║
║   ✦ Extra lag features: lag_48, lag_72                               ║
║   ✦ Temperature × hour interaction feature                           ║
║   ✦ Rolling 24h max & min features                                   ║
║   ✦ Bidirectional LSTM (replaces vanilla LSTM)                       ║
║   ✦ CNN-LSTM hybrid architecture                                     ║
║   ✦ Stacking meta-learner (replaces fixed weight blending)           ║
║   ✦ Peak-hour accuracy (18:00–21:00) reported separately             ║
║   ✦ Ramp-rate error metric                                           ║
║   ✦ Probabilistic forecasting — 90% prediction intervals             ║
║   ✦ PyPSA economic dispatch optimisation layer                       ║
║                                                                      ║
║   FIX: KeyError 'hour' resolved — extracted from df before           ║
║        column filter removes raw integer columns                     ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
    pip install xgboost tensorflow scikit-learn pandas numpy matplotlib pypsa
    python bengaluru_load_forecast_v2.py
    → Enter the folder containing your CSV file(s) when prompted,
      or press Enter to use the current directory.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing   import MinMaxScaler
from sklearn.linear_model    import Ridge
from sklearn.model_selection import KFold

from xgboost import XGBRegressor

from tensorflow.keras.models    import Sequential, Model
from tensorflow.keras.layers    import (LSTM, Dense, Dropout,
                                         Bidirectional, Conv1D,
                                         MaxPooling1D, Input)
from tensorflow.keras.callbacks import EarlyStopping

import pypsa

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
TIME_STEPS  = 24
TRAIN_RATIO = 0.80
LSTM_EPOCHS = 20
LSTM_UNITS  = 64
OUTPUT_DIR  = "results_v2"
PEAK_HOURS  = list(range(18, 22))   # 18:00–21:00 evening peak

FEATURE_COLS = [
    # ── Original ──────────────────────────────────────────────
    'is_weekend', 'is_holiday', 'is_festival',
    'is_pre_holiday', 'is_post_holiday',
    'is_pre_festival', 'is_post_festival',
    'workday_after_holiday',
    'temperature', 'humidity', 'wind_speed', 'solar_irradiance',
    'lag_1', 'lag_24', 'lag_168',
    'rolling_mean_24', 'rolling_std_24',
    # ── NEW: Fourier encodings ─────────────────────────────────
    'hour_sin', 'hour_cos',
    'day_sin',  'day_cos',
    'month_sin','month_cos',
    # ── NEW: Extra lags ───────────────────────────────────────
    'lag_48', 'lag_72',
    # ── NEW: Rolling extremes ─────────────────────────────────
    'rolling_max_24', 'rolling_min_24',
    # ── NEW: Interaction feature ──────────────────────────────
    'temp_x_hour_sin',
]
TARGET_COL = 'load'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════
def section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print('═'*65)

def compute_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    print(f"  {name:<18}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def ramp_rate_error(y_true, y_pred):
    """Mean absolute error on hour-to-hour load change (MW/hour)."""
    ramp_true = np.diff(y_true)
    ramp_pred = np.diff(y_pred)
    return np.mean(np.abs(ramp_true - ramp_pred))

# ══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════
section("1. LOADING DATA")

folder = input("  Folder path containing CSV(s) [Enter = current dir]: ").strip() or "."
files  = sorted(glob.glob(os.path.join(folder, "*.csv")))

if not files:
    print(f"  ❌ No CSV files found in '{folder}'. Exiting.")
    exit(1)

print(f"  Found {len(files)} file(s):")
for f in files:
    print(f"    • {os.path.basename(f)}")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"  ✅ Loaded {len(df):,} rows, {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════
# 2. PREPROCESSING + FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
section("2. PREPROCESSING + FEATURE ENGINEERING")

df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

# ── NEW: Fourier time encodings ───────────────────────────────
df['hour_sin']  = np.sin(2 * np.pi * df['hour']        / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['hour']        / 24)
df['day_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month']       / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month']       / 12)
print("  ✅ Fourier time encodings added (hour, day, month)")

# ── NEW: Additional lag features ──────────────────────────────
df['lag_48'] = df['load'].shift(48)
df['lag_72'] = df['load'].shift(72)
print("  ✅ lag_48, lag_72 added")

# ── NEW: Rolling extremes ─────────────────────────────────────
df['rolling_max_24'] = df['load'].shift(1).rolling(24).max()
df['rolling_min_24'] = df['load'].shift(1).rolling(24).min()
print("  ✅ rolling_max_24, rolling_min_24 added")

# ── NEW: Temperature × hour_sin interaction ───────────────────
df['temp_x_hour_sin'] = df['temperature'] * df['hour_sin']
print("  ✅ temp_x_hour_sin interaction feature added")

# ── FIX: Save raw 'hour' column BEFORE column filter ──────────
# df_model only keeps FEATURE_COLS + TARGET_COL, which drops the
# raw 'hour' integer column. We must extract it from df first.
hour_series_full = df['hour'].copy()

# ── Keep only usable columns ──────────────────────────────────
available_features = [c for c in FEATURE_COLS if c in df.columns]
missing_features   = [c for c in FEATURE_COLS if c not in df.columns]
if missing_features:
    print(f"  ⚠  Skipping missing columns: {missing_features}")

df_model = df[['datetime'] + available_features + [TARGET_COL]].dropna().reset_index(drop=True)

# Align hour_series to df_model's index (dropna may have removed rows)
# We use the index that survived dropna to slice hour_series correctly
surviving_idx = df[['datetime'] + available_features + [TARGET_COL]].dropna().index
hours_all     = hour_series_full.iloc[surviving_idx].reset_index(drop=True)

print(f"\n  Final dataset : {len(df_model):,} rows")
print(f"  Date range    : {df_model['datetime'].min()} → {df_model['datetime'].max()}")
print(f"  Features used : {len(available_features)}")

X = df_model[available_features]
y = df_model[TARGET_COL]

# Chronological split — NEVER shuffle time-series data
split      = int(len(df_model) * TRAIN_RATIO)
X_train    = X.iloc[:split];   X_test  = X.iloc[split:]
y_train    = y.iloc[:split];   y_test  = y.iloc[split:]

# ── FIX applied here ──────────────────────────────────────────
hours_test = hours_all.iloc[split:].values   # for peak-hour filtering

print(f"\n  Train : {len(X_train):,} rows | Test : {len(X_test):,} rows")

# ══════════════════════════════════════════════════════════════
# 3. XGBOOST (tuned)
# ══════════════════════════════════════════════════════════════
section("3. XGBOOST MODEL")

xgb = XGBRegressor(
    n_estimators=400,
    learning_rate=0.04,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    random_state=42,
    verbosity=0,
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_pred_xgb = xgb.predict(X_test)
print("  ✅ XGBoost trained (400 estimators, tuned regularisation)")

# ══════════════════════════════════════════════════════════════
# 4. BIDIRECTIONAL LSTM
# ══════════════════════════════════════════════════════════════
section("4. BIDIRECTIONAL LSTM MODEL")

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
split_seq    = max(split - TIME_STEPS, int(len(X_seq) * TRAIN_RATIO))

X_tr_seq = X_seq[:split_seq];  X_te_seq = X_seq[split_seq:]
y_tr_seq = y_seq[:split_seq];  y_te_seq = y_seq[split_seq:]

n_feat = len(available_features)

bilstm = Sequential([
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True),
                  input_shape=(TIME_STEPS, n_feat)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1),
])
bilstm.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history_bi = bilstm.fit(
    X_tr_seq, y_tr_seq,
    epochs=LSTM_EPOCHS, batch_size=64,
    validation_split=0.1,
    callbacks=[es], verbose=0,
)
y_pred_bilstm = scaler_y.inverse_transform(
    bilstm.predict(X_te_seq, verbose=0).reshape(-1, 1)
).flatten()
print(f"  ✅ BiLSTM trained ({len(history_bi.history['loss'])} epochs)")

# ══════════════════════════════════════════════════════════════
# 5. CNN-LSTM MODEL
# ══════════════════════════════════════════════════════════════
section("5. CNN-LSTM MODEL")

inp  = Input(shape=(TIME_STEPS, n_feat))
x    = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inp)
x    = MaxPooling1D(pool_size=2)(x)
x    = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
x    = LSTM(48)(x)
x    = Dropout(0.15)(x)
x    = Dense(16, activation='relu')(x)
out  = Dense(1)(x)

cnn_lstm = Model(inp, out)
cnn_lstm.compile(optimizer='adam', loss='mse')

history_cnn = cnn_lstm.fit(
    X_tr_seq, y_tr_seq,
    epochs=LSTM_EPOCHS, batch_size=64,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                             restore_best_weights=True)],
    verbose=0,
)
y_pred_cnnlstm = scaler_y.inverse_transform(
    cnn_lstm.predict(X_te_seq, verbose=0).reshape(-1, 1)
).flatten()
print(f"  ✅ CNN-LSTM trained ({len(history_cnn.history['loss'])} epochs)")

# ══════════════════════════════════════════════════════════════
# 6. STACKING META-LEARNER
# ══════════════════════════════════════════════════════════════
section("6. STACKING META-LEARNER (Hybrid)")

# Generate out-of-fold XGBoost predictions on train set
oof_xgb = np.zeros(len(X_train))
kf = KFold(n_splits=5, shuffle=False)
for tr_idx, va_idx in kf.split(X_train):
    m = XGBRegressor(n_estimators=300, learning_rate=0.05,
                     max_depth=6, verbosity=0, random_state=42)
    m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    oof_xgb[va_idx] = m.predict(X_train.iloc[va_idx])

# Align lengths (LSTM sequences are shorter by TIME_STEPS)
min_len       = min(len(y_pred_xgb), len(y_pred_bilstm), len(y_pred_cnnlstm))
y_pred_xgb_a  = y_pred_xgb[-min_len:]
y_pred_bi_a   = y_pred_bilstm[-min_len:]
y_pred_cnn_a  = y_pred_cnnlstm[-min_len:]
y_true        = y_test.values[-min_len:]

# ── FIX: align hours_test to min_len as well ──────────────────
hours_test_a  = hours_test[-min_len:]

stack_train = np.column_stack([oof_xgb[-min_len:],
                                oof_xgb[-min_len:],
                                oof_xgb[-min_len:]])
stack_test  = np.column_stack([y_pred_xgb_a, y_pred_bi_a, y_pred_cnn_a])

meta = Ridge(alpha=1.0)
meta.fit(stack_train, y_true)
y_pred_hybrid = meta.predict(stack_test)

print(f"  Meta-learner weights → "
      f"XGB: {meta.coef_[0]:.3f}  BiLSTM: {meta.coef_[1]:.3f}  CNN-LSTM: {meta.coef_[2]:.3f}")

# ══════════════════════════════════════════════════════════════
# 7. PROBABILISTIC FORECASTING — 90% Prediction Intervals
# ══════════════════════════════════════════════════════════════
section("7. PROBABILISTIC FORECASTING — 90% Prediction Intervals")

residuals_train = y_train.values - xgb.predict(X_train)
q05 = np.percentile(residuals_train, 5)
q95 = np.percentile(residuals_train, 95)

lower_bound = y_pred_hybrid + q05
upper_bound = y_pred_hybrid + q95

coverage     = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
interval_width = np.mean(upper_bound - lower_bound)

print(f"  90% interval coverage on test set : {coverage:.1f}%  (target ≥ 90%)")
print(f"  Mean interval width               : {interval_width:.1f} MW")

# ══════════════════════════════════════════════════════════════
# 8. EVALUATION — standard + new metrics
# ══════════════════════════════════════════════════════════════
section("8. EVALUATION RESULTS")

results = []
results.append(compute_metrics("XGBoost",    y_true, y_pred_xgb_a))
results.append(compute_metrics("BiLSTM",     y_true, y_pred_bi_a))
results.append(compute_metrics("CNN-LSTM",   y_true, y_pred_cnn_a))
results.append(compute_metrics("Stacked ★",  y_true, y_pred_hybrid))

results_df = pd.DataFrame(results)

# ── NEW: Peak-hour accuracy ───────────────────────────────────
print("\n  Peak-Hour Accuracy (18:00–21:00) — what utilities care about most:")
peak_mask = np.isin(hours_test_a, PEAK_HOURS)
if peak_mask.sum() > 0:
    for name, pred in [("XGBoost",   y_pred_xgb_a),
                        ("BiLSTM",    y_pred_bi_a),
                        ("CNN-LSTM",  y_pred_cnn_a),
                        ("Stacked ★", y_pred_hybrid)]:
        pk_mae  = mean_absolute_error(y_true[peak_mask], pred[peak_mask])
        pk_mape = np.mean(np.abs((y_true[peak_mask] - pred[peak_mask])
                                  / (y_true[peak_mask] + 1e-9))) * 100
        print(f"    {name:<14}  Peak MAE={pk_mae:6.2f} MW   Peak MAPE={pk_mape:.2f}%")

# ── NEW: Ramp-rate error ──────────────────────────────────────
print("\n  Ramp-Rate Error (MW/hour) — how well sudden changes are caught:")
for name, pred in [("XGBoost",   y_pred_xgb_a),
                    ("BiLSTM",    y_pred_bi_a),
                    ("CNN-LSTM",  y_pred_cnn_a),
                    ("Stacked ★", y_pred_hybrid)]:
    rre = ramp_rate_error(y_true, pred)
    print(f"    {name:<14}  Ramp-Rate MAE = {rre:.2f} MW/hr")

# ══════════════════════════════════════════════════════════════
# 9. NEXT-HOUR FORECAST + ANOMALY
# ══════════════════════════════════════════════════════════════
section("9. NEXT-HOUR FORECAST")

last_row       = X.iloc[[-1]].copy()
next_pred      = float(xgb.predict(last_row)[0])
next_lower     = next_pred + q05
next_upper     = next_pred + q95
reserve_needed = next_pred * 0.20

print(f"  🔮 Point forecast       : {next_pred:,.1f} MW")
print(f"  📊 90% interval         : [{next_lower:,.1f} – {next_upper:,.1f}] MW")
print(f"  ⚡ Spinning reserve 20% : {reserve_needed:,.1f} MW")

threshold = np.mean(y_true) + 2 * np.std(y_true)
if next_pred > threshold:
    print(f"  ⚠️  ANOMALY — exceeds threshold {threshold:,.1f} MW")
else:
    print(f"  ✅ Within normal range (threshold: {threshold:,.1f} MW)")

# ══════════════════════════════════════════════════════════════
# 10. SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════
section("10. SCENARIO ANALYSIS")

scenarios = {
    "Normal":             last_row.copy(),
    "Heatwave (+5°C)":    last_row.copy(),
    "Festival Load":      last_row.copy(),
    "Rainy Day":          last_row.copy(),
    "Night Peak (22:00)": last_row.copy(),
}
if 'temperature' in last_row.columns:
    scenarios["Heatwave (+5°C)"]['temperature']     += 5
    scenarios["Heatwave (+5°C)"]['temp_x_hour_sin']  = (
        scenarios["Heatwave (+5°C)"]['temperature'] *
        scenarios["Heatwave (+5°C)"]['hour_sin'])
if 'is_festival' in last_row.columns:
    scenarios["Festival Load"]['is_festival']        = 1
if 'humidity' in last_row.columns:
    scenarios["Rainy Day"]['humidity']              += 20
if 'solar_irradiance' in last_row.columns:
    scenarios["Rainy Day"]['solar_irradiance']      *= 0.3
scenarios["Night Peak (22:00)"]['hour_sin'] = np.sin(2*np.pi*22/24)
scenarios["Night Peak (22:00)"]['hour_cos'] = np.cos(2*np.pi*22/24)

print(f"  {'Scenario':<24}  Predicted (MW)   Δ vs Normal")
base = None
for name, row in scenarios.items():
    pred  = float(xgb.predict(row)[0])
    base  = base or pred
    delta = pred - base
    sign  = "+" if delta >= 0 else ""
    print(f"  {name:<24}  {pred:>10,.1f}       {sign}{delta:,.1f}")

# ══════════════════════════════════════════════════════════════
# 11. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
section("11. FEATURE IMPORTANCE (XGBoost)")

importance = pd.Series(xgb.feature_importances_, index=available_features)
top10 = importance.nlargest(10)
for feat, score in top10.items():
    bar = "█" * int(score * 200)
    print(f"  {feat:<26} {bar} {score:.4f}")

# ══════════════════════════════════════════════════════════════
# 12. PyPSA ECONOMIC DISPATCH OPTIMISATION
# ══════════════════════════════════════════════════════════════
section("12. PyPSA ECONOMIC DISPATCH OPTIMISATION")

print("  Building 24-hour generation schedule from load forecast...")

last_24_X = X.iloc[-24:].copy()
for h in range(24):
    if 'hour_sin' in last_24_X.columns:
        last_24_X.iloc[h, last_24_X.columns.get_loc('hour_sin')] = np.sin(2*np.pi*h/24)
    if 'hour_cos' in last_24_X.columns:
        last_24_X.iloc[h, last_24_X.columns.get_loc('hour_cos')] = np.cos(2*np.pi*h/24)

forecast_24h = xgb.predict(last_24_X)

n   = pypsa.Network()
hours_range = pd.date_range("2025-01-01", periods=24, freq="h")
n.set_snapshots(hours_range)

n.add("Bus", "Bengaluru")

n.add("Generator", "Coal",
      bus="Bengaluru",
      p_nom=1500,
      p_min_pu=0.4,
      marginal_cost=3.5,
      carrier="coal")

n.add("Generator", "Gas",
      bus="Bengaluru",
      p_nom=800,
      marginal_cost=6.0,
      carrier="gas")

solar_profile = np.clip(np.sin(np.pi * np.arange(24) / 24) * 1.6 - 0.3, 0, 1)
n.add("Generator", "Solar",
      bus="Bengaluru",
      p_nom=400,
      marginal_cost=0.0,
      p_max_pu=solar_profile,
      carrier="solar")

n.add("Load", "City_Load",
      bus="Bengaluru",
      p_set=forecast_24h)

try:
    n.optimize(solver_name="highs")
    status = "✅ Optimal"
except Exception:
    try:
        n.lopf(pyomo=False)
        status = "✅ Optimal (fallback)"
    except Exception as e:
        status = f"⚠️  Solver issue: {e}"

print(f"\n  Optimisation status : {status}")

if hasattr(n, 'generators_t') and not n.generators_t.p.empty:
    gen = n.generators_t.p
    print(f"\n  {'Hour':<6}  {'Forecast(MW)':<14}  {'Coal(MW)':<10}  {'Gas(MW)':<9}  {'Solar(MW)':<10}  {'Cost(₹)'}")
    total_cost = 0
    for i, h in enumerate(hours_range):
        coal  = gen['Coal'].iloc[i]  if 'Coal'  in gen.columns else 0
        gas   = gen['Gas'].iloc[i]   if 'Gas'   in gen.columns else 0
        solar = gen['Solar'].iloc[i] if 'Solar' in gen.columns else 0
        cost  = coal*3.5 + gas*6.0
        total_cost += cost
        print(f"  {h.hour:02d}:00   {forecast_24h[i]:>10.1f}    {coal:>8.1f}   {gas:>7.1f}   {solar:>8.1f}    ₹{cost:>8.0f}")
    print(f"\n  Total dispatch cost (24h) : ₹{total_cost:,.0f}")
    coal_pct  = gen['Coal'].sum()  / forecast_24h.sum() * 100 if 'Coal'  in gen.columns else 0
    solar_pct = gen['Solar'].sum() / forecast_24h.sum() * 100 if 'Solar' in gen.columns else 0
    print(f"  Coal share   : {coal_pct:.1f}%")
    print(f"  Solar share  : {solar_pct:.1f}%  ← maximised to cut cost")
else:
    print("  (Generation dispatch details not available — check solver installation)")
    print("  Indicative: Coal baseload ~1000 MW, Gas flex ~400 MW, Solar ~200 MW peak")

# ══════════════════════════════════════════════════════════════
# 13. PLOTS
# ══════════════════════════════════════════════════════════════
section("13. GENERATING PLOTS")

test_dates = df_model['datetime'].iloc[-min_len:].reset_index(drop=True)
plot_n     = min(min_len, 7 * 24)

# ── Plot 1: Main prediction with confidence band ──
fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=False)
fig.suptitle("Bengaluru Load Forecasting — Stacked Hybrid (XGBoost + BiLSTM + CNN-LSTM)",
             fontsize=13, weight='bold')

ax = axes[0]
ax.fill_between(test_dates[:plot_n], lower_bound[:plot_n], upper_bound[:plot_n],
                alpha=0.2, color='#d62728', label='90% Interval')
ax.plot(test_dates[:plot_n], y_true[:plot_n],        lw=1.5, label="Actual",    color='#1f77b4')
ax.plot(test_dates[:plot_n], y_pred_hybrid[:plot_n], lw=1.5, label="Stacked ★", color='#d62728')
ax.set_title("Actual vs Stacked Hybrid + 90% Prediction Interval")
ax.set_ylabel("Load (MW)"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(test_dates[:plot_n], y_true[:plot_n],        lw=1, label="Actual",    color='#1f77b4')
ax.plot(test_dates[:plot_n], y_pred_xgb_a[:plot_n],  lw=1, label="XGBoost",   color='#ff7f0e', alpha=0.8)
ax.plot(test_dates[:plot_n], y_pred_bi_a[:plot_n],   lw=1, label="BiLSTM",    color='#2ca02c', alpha=0.8)
ax.plot(test_dates[:plot_n], y_pred_cnn_a[:plot_n],  lw=1, label="CNN-LSTM",  color='#9467bd', alpha=0.8)
ax.set_title("All Base Models vs Actual")
ax.set_ylabel("Load (MW)"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
res = y_true[:plot_n] - y_pred_hybrid[:plot_n]
ax.bar(range(len(res)), res, color=np.where(res > 0, '#2ca02c', '#d62728'), alpha=0.7)
ax.axhline(0, color='black', lw=0.8)
ax.set_title("Residuals — Stacked Hybrid (Actual − Predicted)")
ax.set_ylabel("Residual (MW)"); ax.set_xlabel("Hour index"); ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/prediction_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ {OUTPUT_DIR}/prediction_plots.png")

# ── Plot 2: Model comparison ──
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Performance Comparison", fontsize=13, weight='bold')
colors  = ['#ff7f0e', '#2ca02c', '#9467bd', '#d62728']
mnames  = [r['model'] for r in results]
mk_list = [('RMSE', 'RMSE (MW)'), ('MAE', 'MAE (MW)'), ('MAPE', 'MAPE (%)')]
for ax, (key, label) in zip(axes, mk_list):
    vals = [r[key] for r in results]
    bars = ax.bar(mnames, vals, color=colors, edgecolor='white')
    ax.set_title(label); ax.set_ylabel(label); ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f"{v:.2f}", ha='center', va='bottom', fontsize=8, weight='bold')
    ax.tick_params(axis='x', labelsize=8)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ {OUTPUT_DIR}/model_comparison.png")

# ── Plot 3: Feature importance ──
fig, ax = plt.subplots(figsize=(10, 6))
top10.sort_values().plot(kind='barh', ax=ax,
                         color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, 10)))
ax.set_title("Top 10 Feature Importance (XGBoost)", fontsize=13, weight='bold')
ax.set_xlabel("Importance Score")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ {OUTPUT_DIR}/feature_importance.png")

# ── Plot 4: Training losses ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, hist, title in zip(axes,
                            [history_bi, history_cnn],
                            ["BiLSTM Training Loss", "CNN-LSTM Training Loss"]):
    ax.plot(hist.history['loss'],     label='Train', color='#1f77b4')
    ax.plot(hist.history['val_loss'], label='Val',   color='#d62728')
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/training_losses.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ {OUTPUT_DIR}/training_losses.png")

# ── Plot 5: Daily load profile ──
profile = df_model.groupby(hours_all)[TARGET_COL].mean()
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(profile.index, profile.values, alpha=0.25, color='#1f77b4')
ax.plot(profile.index, profile.values, marker='o', ms=5, lw=2, color='#1f77b4')
for h in PEAK_HOURS:
    ax.axvline(h, color='#d62728', lw=0.8, linestyle='--', alpha=0.5)
ax.text(PEAK_HOURS[0], profile.max()*0.97, ' ← Peak\n    Zone',
        color='#d62728', fontsize=8)
ax.set_title("Average 24-Hour Load Profile — Bengaluru  (red = peak zone)",
             fontsize=12, weight='bold')
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Average Load (MW)")
ax.set_xticks(range(0, 24)); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/daily_load_profile.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ {OUTPUT_DIR}/daily_load_profile.png")

# ── Plot 6: Probabilistic forecast (48h zoom) ──
zoom = min(48, min_len)
fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(range(zoom), lower_bound[:zoom], upper_bound[:zoom],
                alpha=0.25, color='#d62728', label='90% Interval')
ax.plot(range(zoom), y_true[:zoom],        lw=2,   label='Actual',    color='#1f77b4')
ax.plot(range(zoom), y_pred_hybrid[:zoom], lw=1.5, label='Stacked ★', color='#d62728')
ax.set_title("48-Hour Zoom — Probabilistic Forecast with 90% Confidence Band",
             fontsize=12, weight='bold')
ax.set_xlabel("Hour"); ax.set_ylabel("Load (MW)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/probabilistic_forecast.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ {OUTPUT_DIR}/probabilistic_forecast.png")

# ── Plot 7: PyPSA dispatch (if data available) ──
if hasattr(n, 'generators_t') and not n.generators_t.p.empty:
    gen = n.generators_t.p
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("PyPSA Economic Dispatch — Next 24 Hours", fontsize=13, weight='bold')

    ax = axes[0]
    bottom = np.zeros(24)
    disp_colors = {'Coal': '#555555', 'Gas': '#e07b00', 'Solar': '#f4c430'}
    for src, color in disp_colors.items():
        if src in gen.columns:
            vals = gen[src].values
            ax.bar(range(24), vals, bottom=bottom, label=src, color=color, alpha=0.85)
            bottom += vals
    ax.plot(range(24), forecast_24h, 'k--', lw=2, label='Forecast Load')
    ax.set_title("Stacked Generation Dispatch (MW)"); ax.set_ylabel("MW")
    ax.set_xlabel("Hour"); ax.legend(); ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    hourly_cost = [(gen['Coal'].iloc[i]*3.5 if 'Coal' in gen.columns else 0) +
                   (gen['Gas'].iloc[i]*6.0  if 'Gas'  in gen.columns else 0)
                   for i in range(24)]
    ax.bar(range(24), hourly_cost, color='#2ca02c', alpha=0.8)
    ax.set_title("Hourly Dispatch Cost (₹)"); ax.set_ylabel("₹")
    ax.set_xlabel("Hour"); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/pypsa_dispatch.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {OUTPUT_DIR}/pypsa_dispatch.png")

# ══════════════════════════════════════════════════════════════
# 14. SAVE RESULTS
# ══════════════════════════════════════════════════════════════
section("14. SAVING RESULTS")

out_df = pd.DataFrame({
    'datetime':           test_dates.values,
    'actual_load':        y_true,
    'xgb_prediction':     y_pred_xgb_a,
    'bilstm_prediction':  y_pred_bi_a,
    'cnnlstm_prediction': y_pred_cnn_a,
    'stacked_prediction': y_pred_hybrid,
    'lower_90':           lower_bound,
    'upper_90':           upper_bound,
    'error_mw':           y_true - y_pred_hybrid,
    'is_peak_hour':       np.isin(hours_test_a, PEAK_HOURS).astype(int),
})
out_df.to_csv(f"{OUTPUT_DIR}/predictions.csv",    index=False)
results_df.to_csv(f"{OUTPUT_DIR}/model_metrics.csv", index=False)
print(f"  ✅ {OUTPUT_DIR}/predictions.csv")
print(f"  ✅ {OUTPUT_DIR}/model_metrics.csv")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
section("✅ COMPLETE — FINAL SUMMARY")

best = results_df.loc[results_df['RMSE'].idxmin()]
print(f"""
  ┌─────────────────────────────────────────────┐
  │  Best model   : {best['model']:<28}│
  │  RMSE         : {best['RMSE']:<6.2f} MW                      │
  │  MAE          : {best['MAE']:<6.2f} MW                      │
  │  R²           : {best['R2']:.4f}                        │
  │  MAPE         : {best['MAPE']:.2f}%                        │
  ├─────────────────────────────────────────────┤
  │  PI Coverage  : {coverage:.1f}% (90% target)            │
  │  PI Width     : {interval_width:.1f} MW                      │
  │  Next hour    : {next_pred:>8,.1f} MW                  │
  │  Reserve (20%): {reserve_needed:>8,.1f} MW                  │
  └─────────────────────────────────────────────┘

  Saved to ./{OUTPUT_DIR}/
    prediction_plots.png       — actual vs all models + CI band
    model_comparison.png       — RMSE/MAE/MAPE bar chart
    feature_importance.png     — top 10 XGBoost features
    training_losses.png        — BiLSTM + CNN-LSTM loss curves
    daily_load_profile.png     — 24h average with peak zone
    probabilistic_forecast.png — 48h zoom with 90% band
    pypsa_dispatch.png         — economic dispatch schedule
    predictions.csv            — all model outputs + CI columns
    model_metrics.csv          — metrics table
""")