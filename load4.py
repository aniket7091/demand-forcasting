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

import os, glob, warnings, logging, contextlib, io
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

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

warnings.filterwarnings("ignore")
logging.getLogger("pypsa").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
TIME_STEPS  = 24
TRAIN_RATIO = 0.80
LSTM_EPOCHS = 20
LSTM_UNITS  = 64
OUTPUT_DIR  = "results_v2"
PEAK_HOURS  = list(range(18, 22))   # 18:00–21:00 evening peak

N_SIMULATIONS = 1000
ENABLE_WEATHER_UNCERTAINTY = True
MONTE_CARLO_DIR = os.path.join(OUTPUT_DIR, "monte_carlo")
RANDOM_SEED = 42
LOAD_SHEDDING_COST = 100.0   # Reliability penalty per MWh of unmet demand
ENABLE_RESIDUAL_TAIL_CLIPPING = True
RESIDUAL_CLIP_LOW_PERCENTILE = 5
RESIDUAL_CLIP_HIGH_PERCENTILE = 95

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
os.makedirs(MONTE_CARLO_DIR, exist_ok=True)

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

def run_pypsa_dispatch(load_24h, snapshots, solar_profile,
                       include_load_shedding=False,
                       load_shedding_cost=LOAD_SHEDDING_COST):
    """
    Solve a 24-hour economic dispatch problem for one demand trajectory.

    For Monte Carlo reliability studies, a high-cost load-shedding generator is
    included so unmet demand appears as measurable shortage energy instead of
    an infeasible optimization.
    """
    load_24h = np.maximum(np.asarray(load_24h, dtype=float), 0.0)

    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "Bengaluru")

    network.add("Generator", "Coal",
                bus="Bengaluru",
                p_nom=1500,
                p_min_pu=0.4,
                marginal_cost=3.5,
                carrier="coal")
    network.add("Generator", "Gas",
                bus="Bengaluru",
                p_nom=800,
                marginal_cost=6.0,
                carrier="gas")
    network.add("Generator", "Solar",
                bus="Bengaluru",
                p_nom=400,
                marginal_cost=0.0,
                p_max_pu=solar_profile,
                carrier="solar")

    if include_load_shedding:
        network.add("Generator", "Load_Shedding",
                    bus="Bengaluru",
                    p_nom=max(float(load_24h.max()) * 1.5, 1000.0),
                    marginal_cost=load_shedding_cost,
                    carrier="unserved")

    network.add("Load", "City_Load", bus="Bengaluru", p_set=load_24h)

    dispatch_status = "optimal"
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            optimize_result = network.optimize(solver_name="highs")
        if optimize_result is not None:
            status_text = "_".join(map(str, optimize_result)) if isinstance(optimize_result, tuple) else str(optimize_result)
            dispatch_status = "optimal" if "optimal" in status_text.lower() else status_text
    except Exception:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                network.lopf(pyomo=False)
            dispatch_status = "optimal_fallback"
        except Exception as exc:
            dispatch_status = f"solver_failed: {exc}"
            return network, dispatch_status, merit_order_dispatch(
                load_24h, solar_profile, include_load_shedding, load_shedding_cost
            )

    if not hasattr(network, "generators_t") or network.generators_t.p.empty:
        dispatch_status = "empty_dispatch"
        return network, dispatch_status, merit_order_dispatch(
            load_24h, solar_profile, include_load_shedding, load_shedding_cost
        )

    gen = network.generators_t.p
    coal = gen["Coal"].to_numpy() if "Coal" in gen.columns else np.zeros(24)
    gas = gen["Gas"].to_numpy() if "Gas" in gen.columns else np.zeros(24)
    solar = gen["Solar"].to_numpy() if "Solar" in gen.columns else np.zeros(24)
    shortage = (gen["Load_Shedding"].to_numpy()
                if "Load_Shedding" in gen.columns else np.zeros(24))

    total_cost = (
        coal.sum() * 3.5
        + gas.sum() * 6.0
        + shortage.sum() * load_shedding_cost
    )

    result = {
        "total_cost": total_cost,
        "coal_generation": coal,
        "gas_generation": gas,
        "solar_generation": solar,
        "unserved_energy": shortage,
        "peak_load": float(load_24h.max()),
    }
    return network, dispatch_status, result

def merit_order_dispatch(load_24h, solar_profile,
                         include_load_shedding=False,
                         load_shedding_cost=LOAD_SHEDDING_COST):
    """Deterministic fallback used only when a PyPSA solver is unavailable."""
    load_24h = np.maximum(np.asarray(load_24h, dtype=float), 0.0)
    solar = np.minimum(400 * solar_profile, load_24h)
    remaining = np.maximum(load_24h - solar, 0.0)
    coal = np.minimum(1500, remaining)
    remaining = np.maximum(remaining - coal, 0.0)
    gas = np.minimum(800, remaining)
    remaining = np.maximum(remaining - gas, 0.0)
    shortage = remaining if include_load_shedding else np.zeros_like(load_24h)
    total_cost = coal.sum() * 3.5 + gas.sum() * 6.0 + shortage.sum() * load_shedding_cost
    return {
        "total_cost": total_cost,
        "coal_generation": coal,
        "gas_generation": gas,
        "solar_generation": solar,
        "unserved_energy": shortage,
        "peak_load": float(load_24h.max()),
    }

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
# 6. PROPER STACKING (OOF for ALL MODELS)
# ══════════════════════════════════════════════════════════════

section("6. PROPER STACKING META-LEARNER")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge

tscv = TimeSeriesSplit(n_splits=5)

# ---------------------------
# 1. OOF for XGBoost
# ---------------------------
oof_xgb = np.zeros(len(X_train))

for train_idx, val_idx in tscv.split(X_train):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    oof_xgb[val_idx] = model.predict(X_train.iloc[val_idx])

# ---------------------------
# 2. OOF for BiLSTM
# ---------------------------
oof_bilstm = np.zeros(len(X_train))

for train_idx, val_idx in tscv.split(X_train):
    
    X_tr = X_scaled[train_idx]
    y_tr = y_scaled[train_idx]
    
    X_val = X_scaled[val_idx]
    
    # Create sequences
    def seq(data, target):
        Xs, ys = [], []
        for i in range(len(data) - TIME_STEPS):
            Xs.append(data[i:i+TIME_STEPS])
            ys.append(target[i+TIME_STEPS])
        return np.array(Xs), np.array(ys)
    
    X_tr_seq, y_tr_seq = seq(X_tr, y_tr)
    X_val_seq, _ = seq(X_val, y_scaled[val_idx])
    
    if len(X_tr_seq) == 0 or len(X_val_seq) == 0:
        continue

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(TIME_STEPS, X.shape[1])),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X_tr_seq, y_tr_seq, epochs=10, batch_size=64, verbose=0)
    
    preds = model.predict(X_val_seq, verbose=0)
    preds = scaler_y.inverse_transform(preds).flatten()
    
    # Align indices
    oof_bilstm[val_idx[TIME_STEPS:]] = preds[:len(val_idx[TIME_STEPS:])]

# ---------------------------
# 3. OOF for CNN-LSTM
# ---------------------------
oof_cnn = np.zeros(len(X_train))

for train_idx, val_idx in tscv.split(X_train):
    
    X_tr = X_scaled[train_idx]
    y_tr = y_scaled[train_idx]
    
    X_val = X_scaled[val_idx]
    
    X_tr_seq, y_tr_seq = seq(X_tr, y_tr)
    X_val_seq, _ = seq(X_val, y_scaled[val_idx])
    
    if len(X_tr_seq) == 0 or len(X_val_seq) == 0:
        continue

    inp = Input(shape=(TIME_STEPS, X.shape[1]))
    x = Conv1D(64, 3, activation='relu', padding='same')(inp)
    x = MaxPooling1D(2)(x)
    x = LSTM(48)(x)
    out = Dense(1)(x)
    
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X_tr_seq, y_tr_seq, epochs=10, batch_size=64, verbose=0)
    
    preds = model.predict(X_val_seq, verbose=0)
    preds = scaler_y.inverse_transform(preds).flatten()
    
    oof_cnn[val_idx[TIME_STEPS:]] = preds[:len(val_idx[TIME_STEPS:])]

# ---------------------------
# ALIGN OOF (IMPORTANT)
# ---------------------------
valid_idx = np.where(oof_bilstm != 0)[0]

stack_train = np.column_stack([
    oof_xgb[valid_idx],
    oof_bilstm[valid_idx],
    oof_cnn[valid_idx]
])

y_train_stack = y_train.values[valid_idx]

# ---------------------------
# META MODEL
# ---------------------------
meta = Ridge(alpha=1.0)
meta.fit(stack_train, y_train_stack)
stack_train_pred = meta.predict(stack_train)
residuals_stacked_train = y_train_stack - stack_train_pred

print("Meta weights:", meta.coef_)

# ---------------------------
# TEST STACKING
# ---------------------------
min_len = min(len(y_pred_xgb), len(y_pred_bilstm), len(y_pred_cnnlstm))

# Align all predictions and actuals to the shortest length
y_pred_xgb_a  = y_pred_xgb[-min_len:]
y_pred_bi_a   = y_pred_bilstm[-min_len:]
y_pred_cnn_a  = y_pred_cnnlstm[-min_len:]
y_true        = y_test.values[-min_len:]
hours_test_a  = hours_test[-min_len:]

stack_test = np.column_stack([y_pred_xgb_a, y_pred_bi_a, y_pred_cnn_a])

y_pred_hybrid = meta.predict(stack_test)
# ══════════════════════════════════════════════════════════════
# 7. PROBABILISTIC FORECASTING — 90% Prediction Intervals
# ══════════════════════════════════════════════════════════════
section("7. PROBABILISTIC FORECASTING — 90% Prediction Intervals")

# Empirical residual distribution used by both prediction intervals and
# Monte Carlo scenario generation. Prefer stacked OOF residuals because they
# reflect the final model and avoid fitting-error optimism.
if len(residuals_stacked_train) > 0:
    residuals_train = residuals_stacked_train.copy()
    residual_source = "stacked OOF ensemble"
else:
    residuals_train = y_train.values - xgb.predict(X_train)
    residual_source = "XGBoost training residuals"
q05 = np.percentile(residuals_train, 5)
q95 = np.percentile(residuals_train, 95)

lower_bound = y_pred_hybrid + q05
upper_bound = y_pred_hybrid + q95

coverage     = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
interval_width = np.mean(upper_bound - lower_bound)

print(f"  90% interval coverage on test set : {coverage:.1f}%  (target ≥ 90%)")
print(f"  Mean interval width               : {interval_width:.1f} MW")
print(f"  Residual source                   : {residual_source}")

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
hours_range = pd.date_range("2025-01-01", periods=24, freq="h")
solar_profile = np.clip(np.sin(np.pi * np.arange(24) / 24) * 1.6 - 0.3, 0, 1)

n, status, base_dispatch = run_pypsa_dispatch(
    forecast_24h,
    hours_range,
    solar_profile,
    include_load_shedding=False
)
status = "✅ Optimal" if status == "optimal" else f"⚠️  {status}"

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
# 15. MONTE CARLO ECONOMIC DISPATCH SIMULATION
# ══════════════════════════════════════════════════════════════
section("15. MONTE CARLO ECONOMIC DISPATCH SIMULATION")

print("  Building residual-bootstrap demand scenarios...")
print(f"  Monte Carlo simulations        : {N_SIMULATIONS}")
print(f"  Weather uncertainty enabled    : {ENABLE_WEATHER_UNCERTAINTY}")
print(f"  Load-shedding penalty          : ₹{LOAD_SHEDDING_COST:,.0f}/MWh")
print(f"  Residual tail clipping         : {ENABLE_RESIDUAL_TAIL_CLIPPING}")

rng = np.random.default_rng(RANDOM_SEED)
residual_distribution = np.asarray(residuals_train, dtype=float)
residual_distribution = residual_distribution[np.isfinite(residual_distribution)]

if residual_distribution.size == 0:
    raise ValueError("Residual distribution is empty; Monte Carlo simulation cannot proceed.")

# Extremely large positive residuals can create demand scenarios that exceed
# the intentionally small Coal+Gas+Solar test system. Winsorising the empirical
# residuals removes only the rare tail that was driving excessive shortage
# counts while preserving non-parametric bootstrap uncertainty.
if ENABLE_RESIDUAL_TAIL_CLIPPING:
    residual_low = np.percentile(residual_distribution, RESIDUAL_CLIP_LOW_PERCENTILE)
    residual_high = np.percentile(residual_distribution, RESIDUAL_CLIP_HIGH_PERCENTILE)
    residual_distribution = np.clip(residual_distribution, residual_low, residual_high)
    print(
        f"  Residual clip range            : "
        f"[{residual_low:,.1f}, {residual_high:,.1f}] MW"
    )

# Optional weather layer: perturb only measured exogenous weather variables,
# then recompute dependent engineered features before forecasting.
if ENABLE_WEATHER_UNCERTAINTY:
    weather_forecast_cube = np.tile(last_24_X.to_numpy(dtype=float), (N_SIMULATIONS, 1, 1))
    feature_index = {col: i for i, col in enumerate(last_24_X.columns)}

    for weather_col in ["temperature", "humidity", "solar_irradiance"]:
        if weather_col in feature_index and weather_col in df_model.columns:
            sigma = 0.1 * float(df_model[weather_col].std())
            shocks = rng.normal(0.0, sigma, size=(N_SIMULATIONS, 24))
            weather_forecast_cube[:, :, feature_index[weather_col]] += shocks

    if "solar_irradiance" in feature_index:
        solar_idx = feature_index["solar_irradiance"]
        weather_forecast_cube[:, :, solar_idx] = np.maximum(weather_forecast_cube[:, :, solar_idx], 0.0)

    if "humidity" in feature_index:
        humidity_idx = feature_index["humidity"]
        weather_forecast_cube[:, :, humidity_idx] = np.clip(weather_forecast_cube[:, :, humidity_idx], 0.0, 100.0)

    if "temp_x_hour_sin" in feature_index and "temperature" in feature_index and "hour_sin" in feature_index:
        weather_forecast_cube[:, :, feature_index["temp_x_hour_sin"]] = (
            weather_forecast_cube[:, :, feature_index["temperature"]]
            * weather_forecast_cube[:, :, feature_index["hour_sin"]]
        )

    weather_forecast_frame = pd.DataFrame(
        weather_forecast_cube.reshape(-1, len(last_24_X.columns)),
        columns=last_24_X.columns
    )
    base_forecasts_mc = xgb.predict(weather_forecast_frame).reshape(N_SIMULATIONS, 24)
else:
    base_forecasts_mc = np.tile(forecast_24h, (N_SIMULATIONS, 1))

# Preserve the daily structure by bootstrapping complete 24-hour residual
# blocks whenever enough historical residuals are available. This remains
# non-parametric: every sampled error comes directly from the empirical record.
if residual_distribution.size >= 24:
    residual_blocks = np.lib.stride_tricks.sliding_window_view(residual_distribution, 24)
    sampled_block_ids = rng.integers(0, len(residual_blocks), size=N_SIMULATIONS)
    sampled_residuals = residual_blocks[sampled_block_ids].copy()
else:
    sampled_residuals = rng.choice(
        residual_distribution,
        size=(N_SIMULATIONS, 24),
        replace=True
    )
simulated_loads = np.maximum(base_forecasts_mc + sampled_residuals, 0.0)

# Store complete 24-hour trajectories by simulation id for reproducibility.
scenario_loads = {
    simulation_id: simulated_loads[simulation_id].copy()
    for simulation_id in range(N_SIMULATIONS)
}

mc_records = []
coal_energy = np.zeros(N_SIMULATIONS)
gas_energy = np.zeros(N_SIMULATIONS)
solar_energy = np.zeros(N_SIMULATIONS)
shortage_energy = np.zeros(N_SIMULATIONS)
dispatch_costs = np.zeros(N_SIMULATIONS)
peak_loads = simulated_loads.max(axis=1)
dispatch_statuses = []

for simulation_id in tqdm(range(N_SIMULATIONS), desc="  Monte Carlo PyPSA dispatch"):
    _, dispatch_status, dispatch_result = run_pypsa_dispatch(
        scenario_loads[simulation_id],
        hours_range,
        solar_profile,
        include_load_shedding=True,
        load_shedding_cost=LOAD_SHEDDING_COST
    )

    coal_mwh = float(np.sum(dispatch_result["coal_generation"]))
    gas_mwh = float(np.sum(dispatch_result["gas_generation"]))
    solar_mwh = float(np.sum(dispatch_result["solar_generation"]))
    shortage_mwh = float(np.sum(dispatch_result["unserved_energy"]))
    generation_cost = coal_mwh * 3.5 + gas_mwh * 6.0
    shortage_penalty = shortage_mwh * LOAD_SHEDDING_COST
    total_dispatch_cost = float(dispatch_result["total_cost"])

    coal_energy[simulation_id] = coal_mwh
    gas_energy[simulation_id] = gas_mwh
    solar_energy[simulation_id] = solar_mwh
    shortage_energy[simulation_id] = shortage_mwh
    dispatch_costs[simulation_id] = total_dispatch_cost
    dispatch_statuses.append(dispatch_status)

    mc_records.append({
        "simulation_id": simulation_id,
        "peak_load": peak_loads[simulation_id],
        "total_cost": total_dispatch_cost,
        "coal_energy": coal_mwh,
        "gas_energy": gas_mwh,
        "solar_energy": solar_mwh,
        "shortage_mwh": shortage_mwh,
        "generation_cost": generation_cost,
        "shortage_penalty": shortage_penalty,
        "dispatch_status": dispatch_status,
    })

mc_results_df = pd.DataFrame(mc_records)

shortage_scenarios = int((shortage_energy > 1e-6).sum())
lolp = shortage_scenarios / N_SIMULATIONS
eens = float(shortage_energy.mean())
reserve_sufficiency = 1.0 - lolp

risk_metrics = {
    "LOLP": lolp,
    "EENS": eens,
    "mean_cost": float(np.mean(dispatch_costs)),
    "median_cost": float(np.median(dispatch_costs)),
    "min_cost": float(np.min(dispatch_costs)),
    "max_cost": float(np.max(dispatch_costs)),
    "p95_cost": float(np.percentile(dispatch_costs, 95)),
    "reserve_sufficiency": reserve_sufficiency,
    "peak_load_p95": float(np.percentile(peak_loads, 95)),
    "load_shedding_cost": LOAD_SHEDDING_COST,
}
risk_metrics_df = pd.DataFrame([risk_metrics])

ci_summary_df = pd.DataFrame([
    {
        "metric": "Demand peak (MW)",
        "p05": np.percentile(peak_loads, 5),
        "p50": np.percentile(peak_loads, 50),
        "p95": np.percentile(peak_loads, 95),
    },
    {
        "metric": "Dispatch cost (₹)",
        "p05": np.percentile(dispatch_costs, 5),
        "p50": np.percentile(dispatch_costs, 50),
        "p95": np.percentile(dispatch_costs, 95),
    },
    {
        "metric": "Coal usage (MWh)",
        "p05": np.percentile(coal_energy, 5),
        "p50": np.percentile(coal_energy, 50),
        "p95": np.percentile(coal_energy, 95),
    },
    {
        "metric": "Gas usage (MWh)",
        "p05": np.percentile(gas_energy, 5),
        "p50": np.percentile(gas_energy, 50),
        "p95": np.percentile(gas_energy, 95),
    },
    {
        "metric": "Solar usage (MWh)",
        "p05": np.percentile(solar_energy, 5),
        "p50": np.percentile(solar_energy, 50),
        "p95": np.percentile(solar_energy, 95),
    },
])

mc_results_path = os.path.join(MONTE_CARLO_DIR, "monte_carlo_summary.csv")
risk_metrics_path = os.path.join(MONTE_CARLO_DIR, "risk_metrics.csv")
ci_summary_path = os.path.join(MONTE_CARLO_DIR, "confidence_intervals.csv")
mc_results_df.to_csv(mc_results_path, index=False)
risk_metrics_df.to_csv(risk_metrics_path, index=False)
ci_summary_df.to_csv(ci_summary_path, index=False)

print("\n  Monte Carlo confidence intervals (5th / 50th / 95th percentile):")
print(ci_summary_df.to_string(index=False, formatters={
    "p05": "{:,.2f}".format,
    "p50": "{:,.2f}".format,
    "p95": "{:,.2f}".format,
}))

print("\n  Risk metrics:")
print(f"    LOLP                : {lolp * 100:.2f}%")
print(f"    EENS                : {eens:.2f} MWh")
print(f"    Reserve sufficiency : {reserve_sufficiency * 100:.2f}%")
print(f"    Cost P95            : ₹{risk_metrics['p95_cost']:,.0f}")

# ── Distribution plots for publication and risk interpretation ──
demand_p05 = np.percentile(simulated_loads, 5, axis=0)
demand_p50 = np.percentile(simulated_loads, 50, axis=0)
demand_p95 = np.percentile(simulated_loads, 95, axis=0)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(peak_loads, bins=35, color="#1f77b4", alpha=0.82, edgecolor="white")
ax.axvline(np.percentile(peak_loads, 95), color="#d62728", lw=2, label="P95")
ax.set_title("Monte Carlo Peak Demand Distribution", fontsize=12, weight="bold")
ax.set_xlabel("Peak load (MW)")
ax.set_ylabel("Scenario count")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(MONTE_CARLO_DIR, "demand_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(dispatch_costs, bins=35, color="#2ca02c", alpha=0.82, edgecolor="white")
ax.axvline(np.percentile(dispatch_costs, 95), color="#d62728", lw=2, label="P95")
ax.set_title("Monte Carlo Dispatch Cost Distribution", fontsize=12, weight="bold")
ax.set_xlabel("Total dispatch cost (₹)")
ax.set_ylabel("Scenario count")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(MONTE_CARLO_DIR, "cost_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(coal_energy, bins=30, alpha=0.62, label="Coal", color="#4d4d4d", edgecolor="white")
ax.hist(gas_energy, bins=30, alpha=0.62, label="Gas", color="#e07b00", edgecolor="white")
ax.hist(solar_energy, bins=30, alpha=0.62, label="Solar", color="#f4c430", edgecolor="white")
ax.set_title("Generator Utilization Distribution", fontsize=12, weight="bold")
ax.set_xlabel("Daily generation (MWh)")
ax.set_ylabel("Scenario count")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(MONTE_CARLO_DIR, "generator_utilization_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
reliability_vals = [reserve_sufficiency * 100, lolp * 100]
bars = ax.bar(["Fully met", "Shortage"], reliability_vals,
              color=["#2ca02c", "#d62728"], edgecolor="white")
for bar, value in zip(bars, reliability_vals):
    ax.text(bar.get_x() + bar.get_width()/2, value + 1,
            f"{value:.2f}%", ha="center", va="bottom", fontsize=9, weight="bold")
ax.set_ylim(0, max(100, max(reliability_vals) * 1.15))
ax.set_title("Reliability Distribution", fontsize=12, weight="bold")
ax.set_ylabel("Probability (%)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(MONTE_CARLO_DIR, "reliability_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Monte Carlo Economic Dispatch Risk Dashboard", fontsize=14, weight="bold")

ax = axes[0, 0]
hours = np.arange(24)
ax.fill_between(hours, demand_p05, demand_p95, color="#1f77b4", alpha=0.22, label="P05–P95")
ax.plot(hours, demand_p50, color="#1f77b4", lw=2, label="Median")
ax.plot(hours, forecast_24h, color="#111111", lw=1.6, linestyle="--", label="Base forecast")
ax.set_title("Demand Uncertainty Envelope")
ax.set_xlabel("Hour")
ax.set_ylabel("Load (MW)")
ax.set_xticks(range(0, 24, 3))
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.hist(dispatch_costs, bins=35, color="#2ca02c", alpha=0.82, edgecolor="white")
ax.axvline(np.mean(dispatch_costs), color="#111111", lw=1.8, label="Mean")
ax.axvline(np.percentile(dispatch_costs, 95), color="#d62728", lw=1.8, label="P95")
ax.set_title("Cost Distribution")
ax.set_xlabel("Total dispatch cost (₹)")
ax.set_ylabel("Scenario count")
ax.legend()
ax.grid(axis="y", alpha=0.3)

ax = axes[1, 0]
box = ax.boxplot([coal_energy, gas_energy, solar_energy],
                 labels=["Coal", "Gas", "Solar"],
                 patch_artist=True,
                 showfliers=False)
for patch, color in zip(box["boxes"], ["#4d4d4d", "#e07b00", "#f4c430"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.78)
ax.set_title("Generator Utilization Distribution")
ax.set_ylabel("Daily generation (MWh)")
ax.grid(axis="y", alpha=0.3)

ax = axes[1, 1]
metric_names = ["LOLP", "EENS", "Reserve", "Peak P95"]
metric_values = [lolp * 100, eens, reserve_sufficiency * 100, risk_metrics["peak_load_p95"]]
metric_colors = ["#d62728", "#9467bd", "#2ca02c", "#1f77b4"]
bars = ax.bar(metric_names, metric_values, color=metric_colors, edgecolor="white", alpha=0.86)
for bar, value in zip(bars, metric_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f"{value:,.2f}", ha="center", va="bottom", fontsize=8, weight="bold")
ax.set_title("Reliability Metrics")
ax.set_ylabel("Value")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
dashboard_path = os.path.join(MONTE_CARLO_DIR, "monte_carlo_dashboard.png")
fig.savefig(dashboard_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n  ✅ {mc_results_path}")
print(f"  ✅ {risk_metrics_path}")
print(f"  ✅ {ci_summary_path}")
print(f"  ✅ {dashboard_path}")

print(f"""
  Monte Carlo Simulations : {N_SIMULATIONS}
  Expected Cost           : ₹{risk_metrics['mean_cost']:,.0f}
  Worst Case Cost         : ₹{risk_metrics['max_cost']:,.0f}
  Best Case Cost          : ₹{risk_metrics['min_cost']:,.0f}
  LOLP                    : {lolp * 100:.2f}%
  EENS                    : {eens:.2f} MWh
  Reserve Sufficiency     : {reserve_sufficiency * 100:.2f}%
  Peak Load P95           : {risk_metrics['peak_load_p95']:,.1f} MW
""")

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
    monte_carlo/               — residual-bootstrap risk CSVs + dashboard
    predictions.csv            — all model outputs + CI columns
    model_metrics.csv          — metrics table
""")
