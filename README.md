# Demand Dynamic Pricing ML System

## 📊 Project Overview
A comprehensive machine learning system for retail demand forecasting and dynamic pricing optimization. Built to handle large-scale retail data (10M+ transactions) with a focus on promotion effectiveness and inventory management.

## 🎯 Business Problem
Retailers face two critical challenges:
- **Stockouts**: Lost sales when demand exceeds inventory
- **Excess Inventory**: Wasted capital and storage costs

This model predicts daily unit sales at store-item level, enabling optimal inventory management and dynamic pricing decisions.

## 🏆 Key Achievements
- **41% improvement** over baseline (median) predictions
- **Final RMSLE**: 0.52 on holdout data
- **30+ engineered features** including time-based, lag, rolling statistics, and promotion memory
- **Store tiering system** (A/B/C) based on sales volume
- **SHAP analysis** for model explainability

## 📈 Model Performance

| Model | Validation RMSLE | Holdout RMSLE | Improvement |
|-------|-----------------|---------------|-------------|
| Median Baseline | 0.8833 | 0.8833 | Reference |
| SGD Ridge | 0.6234 | - | 29.4% |
| Random Forest | 0.5432 | 0.5721 | 35.2% |
| **XGBoost + Store Tiers** | **0.5043** | **0.5189** | **41.3%** |

**Best Model**: XGBoost with store tiering and promotion buffer

## 🔧 Features Engineered

### Time-Based Features
- Day of week, month, quarter, weekend indicators
- Month start/end, quarter end, year end flags

### Lag Features (Historical Sales)
- 1, 7, 14, 28 day lags

### Rolling Statistics
- Mean and standard deviation for 7, 14, 28 day windows

### Promotion Features
- Promotion lags (1, 7 days)
- Days since last promotion

### Zero-Sale Patterns
- Zero-sale indicators and rolling counts
- Zero-sale ratios for 7 and 28 day windows

### Store Tiering
- Tier A (High volume): Top 25% stores
- Tier B (Medium volume): Middle 50% stores
- Tier C (Low volume): Bottom 25% stores

### Outlier Detection
- 3x multiplier detection based on rolling mean


demand-dynamic-pricing-ml-system/
├── main.py # Main execution script
├── notebooks/ # Jupyter notebooks with analysis
├── .gitignore # Git ignore rules
└── README.md # This file



## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap


# Clone repository
git clone https://github.com/hari54haran98/demand-dynamic-pricing-ml-system.git
cd demand-dynamic-pricing-ml-system

# Install dependencies
pip install -r requirements.txt

Installation
bash
# Clone repository
git clone https://github.com/hari54haran98/demand-dynamic-pricing-ml-system.git
cd demand-dynamic-pricing-ml-system

# Install dependencies
pip install -r requirements.txt
Data Setup
Download train.csv (the dataset is not included in the repository due to size)

Place it in the project root folder

Run the analysis

Run the Analysis
bash
python main.py
📊 Results Visualization
Key visualizations generated:

Daily sales trends and seasonality

Promotion lift analysis

Store-level demand distribution

Item-level demand patterns

Feature importance charts

Error analysis by segment

🧠 Model Explainability (SHAP Analysis)
Top 10 Most Important Features:

sales_lag_1 - Yesterday's sales (32.4% importance)

sales_roll_mean_7 - 7-day moving average (18.3%)

sales_lag_7 - Sales from last week (14.2%)

onpromotion - Promotion flag (9.8%)

store_tier_A - High-volume store indicator (6.7%)

zero_ratio_last_7 - Recent zero-sale pattern (5.4%)

month - Seasonal patterns (4.1%)

day_of_week - Day-of-week effect (3.8%)

is_weekend - Weekend indicator (2.7%)

days_since_last_promo - Promotion recency (1.6%)

💼 Business Impact
Inventory Management
Best prediction days: Tuesday-Thursday (error < 8.5 units)

Risk days: Weekends (error > 12 units)

Recommendation: Keep 20% safety stock on weekends

Promotion Strategy
Promotions increase sales by 30-40% on average

Model under-predicts on promotion days by 12%

Action: Added 30% buffer for promotion day forecasts

Store Segmentation
Tier A stores (high volume): 2x prediction error vs Tier C

Action: Separate models for each tier recommended

📊 Error Analysis
By Store Tier
Tier	MAE (units)	Under-prediction	Over-prediction
A (High)	18.3	42%	48%
B (Medium)	8.7	48%	47%
C (Low)	3.2	52%	43%
By Promotion Status
Condition	MAE	Under-prediction
Non-promotion	7.2	46%
Promotion	14.5	58%
🔮 Future Improvements
Add external features (holidays, weather data)

Implement deep learning models (LSTM, Prophet)

Build real-time prediction API

A/B testing framework for dynamic pricing

Separate models for promotion vs non-promotion days

🛠️ Tech Stack
Python 3.8+ - Core language

pandas, numpy - Data processing

scikit-learn - Baseline models and preprocessing

XGBoost - Primary ML model

SHAP - Model explainability

matplotlib, seaborn - Visualization

📝 License
MIT License - Free for academic and commercial use

👤 Author
GitHub: @hari54haran98

🙏 Acknowledgments
Retail dataset used for demand forecasting

XGBoost library for gradient boosting

SHAP community for explainable AI tools



## 📁 Project Structure
