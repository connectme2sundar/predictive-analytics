import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predictive Analytics Dashboard", page_icon="🔮", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.4rem; font-weight: bold; color: #0f172a; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
SAMPLE_CSV = BASE_DIR / "sample_time_series_data.csv"

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    months = 48
    dates = pd.date_range(start='2022-01-01', periods=months, freq='MS')
    t = np.arange(months)
    trend = 50 + 2.5 * t
    seasonality = 20 * np.sin(2 * np.pi * t / 12) + 10 * np.cos(2 * np.pi * t / 6)
    holiday = np.zeros(months)
    for i in range(months):
        if dates[i].month in [11, 12]:
            holiday[i] = 35 + np.random.normal(0, 5)
        elif dates[i].month in [6, 7]:
            holiday[i] = -10 + np.random.normal(0, 3)
    noise = np.random.normal(0, 8, months)
    sales = trend + seasonality + holiday + noise
    sales = np.clip(sales, 20, None)

    df = pd.DataFrame({
        'date': dates,
        'year': dates.year,
        'month': dates.month,
        'sales': np.round(sales, 2),
        'marketing_spend': np.round(5000 + 50*t + np.random.normal(0, 300, months), 2),
        'website_traffic': np.round(10000 + 200*t + 500*seasonality + np.random.normal(0, 500, months), 0).astype(int),
        'competitor_activity': np.round(np.random.uniform(0, 10, months), 1),
        'economic_index': np.round(100 + 0.5*t + np.random.normal(0, 2, months), 2)
    })
    df['website_traffic'] = df['website_traffic'].clip(lower=5000)
    return df

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

st.sidebar.title("📁 Data Import")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Model Settings")
model_choice = st.sidebar.selectbox("Model", ["Auto (Best)", "Linear Regression", "Random Forest", "Gradient Boosting"])
forecast_months = st.sidebar.slider("Forecast Months", 1, 24, 12)
test_months = st.sidebar.slider("Test Months", 3, 18, 12)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    source = f"Uploaded: {uploaded_file.name}"
elif SAMPLE_CSV.exists():
    df = pd.read_csv(SAMPLE_CSV)
    df['date'] = pd.to_datetime(df['date'])
    source = "sample_time_series_data.csv"
else:
    df = generate_sample_data()
    source = "Auto-generated sample data"

st.sidebar.info(f"Using: {source}")

# Auto-detect columns
def auto_detect_columns(df):
    col_map = {}
    used = set(df.columns.tolist())
    for col in df.columns:
        cl = col.lower().replace(' ', '_')
        target = None
        if 'date' in cl:
            target = 'date'
        elif any(x in cl for x in ['sales', 'revenue', 'target', 'demand']):
            target = 'sales'
        elif any(x in cl for x in ['marketing', 'ad_spend', 'promo']):
            target = 'marketing_spend'
        elif any(x in cl for x in ['traffic', 'visits', 'users']):
            target = 'website_traffic'
        elif any(x in cl for x in ['competitor', 'competition']):
            target = 'competitor_activity'
        elif any(x in cl for x in ['economic', 'gdp', 'index']):
            target = 'economic_index'
        elif 'year' in cl:
            target = 'year'
        elif 'month' in cl:
            target = 'month'

        if target and target != col and target not in used:
            col_map[col] = target
            used.add(target)
    return col_map

col_map = auto_detect_columns(df)
if col_map:
    df = df.rename(columns=col_map)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

if 'sales' not in df.columns:
    st.error("Target column 'sales' not found. Please upload data with a sales/revenue column.")
    st.stop()

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def create_features(df_input):
    d = df_input.copy()
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['quarter'] = d['month'].apply(lambda x: (x-1)//3 + 1)
    d['time_index'] = np.arange(len(d))
    for lag in [1, 2, 3]:
        d[f'sales_lag_{lag}'] = d['sales'].shift(lag)
    d['sales_roll_mean_3'] = d['sales'].shift(1).rolling(window=3).mean()
    if 'marketing_spend' in d.columns and 'economic_index' in d.columns:
        d['marketing_x_economic'] = d['marketing_spend'] * d['economic_index'] / 100000
    return d

df_features = create_features(df)

# Define feature columns (exclude target and date)
feature_cols = [c for c in df_features.columns if c not in ['date', 'sales']]

# Drop rows with NaN from feature engineering
model_df = df_features.dropna().reset_index(drop=True)

# After dropping NaN, recompute feature_cols from actual remaining columns
feature_cols = [c for c in feature_cols if c in model_df.columns]

if len(feature_cols) < 2:
    st.error(f"Not enough valid features after preprocessing. Found: {feature_cols}")
    st.stop()

if len(model_df) <= test_months + 3:
    st.error(f"Not enough data. Need at least {test_months + 4} rows, got {len(model_df)}.")
    st.stop()

train_size = len(model_df) - test_months
train_df = model_df.iloc[:train_size].copy()
test_df = model_df.iloc[train_size:].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df['sales'].copy()
X_test = test_df[feature_cols].copy()
y_test = test_df['sales'].copy()

# ============================================================
# MODEL TRAINING
# ============================================================
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100
    r2 = r2_score(y_true, y_pred)
    return {'Model': model_name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2), 'R2': round(r2, 3)}

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    results.append(evaluate(y_test, pred, name))

results_df = pd.DataFrame(results).sort_values('MAE')

if model_choice == "Auto (Best)":
    best_name = results_df.iloc[0]['Model']
else:
    best_name = model_choice

best_model = models[best_name]
best_pred = predictions[best_name]

# Feature importance (from Random Forest)
rf_model = models['Random Forest']
importance = pd.DataFrame({'feature': feature_cols, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)

# ============================================================
# FORECAST FUTURE - BULLETPROOF VERSION
# ============================================================
last_date = df['date'].max()
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')

# Start with a copy of the last known row for each future month
last_row = df.iloc[-1].copy()

# Build future rows iteratively
future_rows = []
prev_sales = [df['sales'].iloc[-1], df['sales'].iloc[-2], df['sales'].iloc[-3]]

for i in range(forecast_months):
    future_date = future_dates[i]
    row = {
        'date': future_date,
        'year': future_date.year,
        'month': future_date.month,
        'sales': np.nan,  # to be predicted
    }

    # Project known features forward
    if 'marketing_spend' in df.columns:
        row['marketing_spend'] = last_row['marketing_spend'] * (1.02 ** (i+1)) + np.random.normal(0, 100)
    if 'website_traffic' in df.columns:
        row['website_traffic'] = last_row['website_traffic'] * (1.015 ** (i+1)) + np.random.normal(0, 300)
    if 'competitor_activity' in df.columns:
        row['competitor_activity'] = np.clip(last_row['competitor_activity'] + np.random.normal(0, 1), 0, 10)
    if 'economic_index' in df.columns:
        row['economic_index'] = last_row['economic_index'] + 0.5 * (i+1) + np.random.normal(0, 1)

    # Add engineered features
    row['month_sin'] = np.sin(2 * np.pi * row['month'] / 12)
    row['month_cos'] = np.cos(2 * np.pi * row['month'] / 12)
    row['quarter'] = (row['month'] - 1) // 3 + 1
    row['time_index'] = len(df) + i

    # Lag features from previous predictions
    row['sales_lag_1'] = prev_sales[0]
    row['sales_lag_2'] = prev_sales[1]
    row['sales_lag_3'] = prev_sales[2]
    row['sales_roll_mean_3'] = np.mean(prev_sales[:3])

    # Interaction feature
    if 'marketing_spend' in row and 'economic_index' in row:
        row['marketing_x_economic'] = row['marketing_spend'] * row['economic_index'] / 100000

    # Build prediction input with EXACT same columns as training
    X_f = pd.DataFrame([[row.get(c, 0) for c in feature_cols]], columns=feature_cols)

    # Predict
    pred = best_model.predict(X_f)[0]
    row['sales'] = max(pred, 0)

    # Update prev_sales for next iteration
    prev_sales = [row['sales'], prev_sales[0], prev_sales[1]]

    future_rows.append(row)

forecast_df = pd.DataFrame(future_rows)

# ============================================================
# DASHBOARD UI
# ============================================================
st.markdown('<div class="main-header">🔮 Predictive Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Time-Series Forecasting with Machine Learning</div>', unsafe_allow_html=True)

# KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Best Model", best_name)
with kpi2:
    best_mae = results_df[results_df['Model'] == best_name]['MAE'].values[0]
    st.metric("Test MAE", f"{best_mae:.2f}")
with kpi3:
    best_mape = results_df[results_df['Model'] == best_name]['MAPE'].values[0]
    st.metric("Test MAPE", f"{best_mape:.1f}%")
with kpi4:
    best_r2 = results_df[results_df['Model'] == best_name]['R2'].values[0]
    st.metric("R2 Score", f"{best_r2:.3f}")

st.markdown("---")

# Model Comparison
st.subheader("📋 Model Comparison")
st.dataframe(results_df.set_index('Model'), use_container_width=True)

st.markdown("---")

# Charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Sales Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['sales'], mode='lines', name='Historical', line=dict(color='#636EFA', width=2)))
    fig.add_trace(go.Scatter(x=test_df['date'], y=best_pred, mode='lines', name='Predicted (Test)', line=dict(color='#00CC96', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['sales'], mode='lines+markers', name='Forecast', line=dict(color='#EF553B', width=3), marker=dict(size=8)))
    fig.update_layout(template='plotly_white', height=400, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02), yaxis_title='Sales', xaxis_title='Date')
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("🎯 Actual vs Predicted (Test)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df['date'], y=y_test, mode='lines+markers', name='Actual', line=dict(color='#636EFA', width=2)))
    fig.add_trace(go.Scatter(x=test_df['date'], y=best_pred, mode='lines+markers', name='Predicted', line=dict(color='#EF553B', width=2, dash='dash')))
    fig.update_layout(template='plotly_white', height=400, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02), yaxis_title='Sales', xaxis_title='Date')
    st.plotly_chart(fig, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("📊 Residuals")
    residuals = y_test - best_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df['date'], y=residuals, mode='lines+markers', name='Residuals', line=dict(color='#AB63FA', width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(template='plotly_white', height=350, margin=dict(l=20, r=20, t=30, b=20), yaxis_title='Residual', xaxis_title='Date')
    st.plotly_chart(fig, use_container_width=True)

with chart_col4:
    st.subheader("🔑 Feature Importance")
    fig = px.bar(importance.head(10), x='importance', y='feature', orientation='h', template='plotly_white', labels={'importance': 'Importance', 'feature': ''})
    fig.update_layout(height=350, margin=dict(l=150, r=20, t=30, b=20), yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Forecast Table
st.subheader("📅 Forecast Table")
forecast_display = forecast_df[['date', 'sales']].copy()
forecast_display['sales'] = forecast_display['sales'].round(2)
forecast_display.columns = ['Date', 'Predicted Sales']
st.dataframe(forecast_display, use_container_width=True, height=300)

# Downloads
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    csv = forecast_display.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast (CSV)", csv, "sales_forecast.csv", "text/csv")

with col_dl2:
    results_csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Model Results (CSV)", results_csv, "model_results.csv", "text/csv")

st.markdown("---")
st.caption("Built with Streamlit, Plotly & Scikit-Learn | Internship Project - Predictive Analytics")
