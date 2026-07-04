# Predictive Analytics Dashboard

An interactive time-series forecasting dashboard using machine learning to predict future sales trends.

## 🚀 Features

- **Multi-Model Comparison**: Linear Regression, Random Forest, and Gradient Boosting
- **Auto Model Selection**: Automatically picks the best model by MAE
- **Feature Engineering**: Seasonal encoding, lag features, rolling statistics, interaction terms
- **Interactive Forecasting**: Adjustable forecast period (1-24 months) and test split
- **Model Evaluation**: MAE, RMSE, MAPE, and R² scores
- **Residual Analysis**: Visualize prediction errors over time
- **Feature Importance**: Understand which variables drive predictions
- **Data Upload**: Supports CSV and Excel with auto column detection
- **Auto-Generated Data**: Works out-of-the-box with synthetic 48-month dataset

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🎯 Run the Dashboard

```bash
streamlit run app_predictive.py
```

## 📁 Project Files

| File | Description |
|------|-------------|
| `app_predictive.py` | Main Streamlit application |
| `sample_time_series_data.csv` | 48 months of synthetic sales data with features |
| `predictive_analytics_dashboard.html` | Static interactive HTML preview |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## 📊 Data Format

The app auto-detects common column names. Include these for best results:

| Column | Alternative Names |
|--------|-------------------|
| `date` | Date, Time, Period |
| `sales` | Sales, Revenue, Target, Demand |
| `marketing_spend` | Marketing, Ad Spend, Promo |
| `website_traffic` | Traffic, Visits, Users |
| `competitor_activity` | Competitor, Competition |
| `economic_index` | Economic, GDP, Index |
| `year` | Year |
| `month` | Month |

## 🧠 Models Used

| Model | Strengths | Best For |
|-------|-----------|----------|
| **Linear Regression** | Interpretable, fast, no overfitting | Data with clear linear trends |
| **Random Forest** | Handles non-linearity, feature importance | Complex feature interactions |
| **Gradient Boosting** | High accuracy, handles outliers | Noisy data with many features |

## 📈 Forecasting Methodology

1. **Data Preprocessing**: Handle missing values, clip outliers, normalize features
2. **Feature Engineering**:
   - Time features: month sin/cos encoding, quarter, time index
   - Lag features: sales from 1-3 months ago
   - Rolling statistics: 3-month moving average
   - Interactions: marketing × economic index
3. **Train/Test Split**: Last N months as test set (configurable)
4. **Model Training**: Fit all 3 models on training data
5. **Evaluation**: Compare MAE, RMSE, MAPE, R² on test set
6. **Forecast**: Iteratively predict future months using lag features

## 🎯 Internship Learning Outcomes

By completing this project, you will demonstrate:

1. **Predictive Modeling**: Building and comparing multiple regression models
2. **Time-Series Analysis**: Handling trends, seasonality, and autocorrelation
3. **Feature Engineering**: Creating lag, rolling, and interaction features
4. **Model Evaluation**: Interpreting MAE, RMSE, MAPE, and R² metrics
5. **Data Preprocessing**: Cleaning, transforming, and preparing data for ML
6. **Interactive Dashboards**: Building adjustable forecasting tools

## 💡 Extension Ideas

- **ARIMA/SARIMA**: Add classical time-series models for comparison
- **Prophet**: Integrate Facebook Prophet for automatic seasonality detection
- **Hyperparameter Tuning**: Add GridSearchCV for model optimization
- **Confidence Intervals**: Show prediction intervals on forecasts
- **What-If Analysis**: Sliders to simulate marketing spend changes
- **Multi-Step Evaluation**: Evaluate forecasts at 1, 3, 6, 12 month horizons
- **Anomaly Detection**: Flag unusual sales patterns automatically
- **Export to Excel**: Generate formatted forecast reports

## 🛠️ Tech Stack

- **ML**: scikit-learn (LinearRegression, RandomForest, GradientBoosting)
- **Visualization**: Plotly
- **Dashboard**: Streamlit
- **Data**: Pandas, NumPy

## 📄 License

Created for educational and internship purposes.
