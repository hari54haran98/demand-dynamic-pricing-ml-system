# Demand Dynamic Pricing ML System

## 📊 Project Overview
A comprehensive machine learning system for retail demand forecasting and dynamic pricing optimization. Built to handle large-scale retail data (10.4M+ transactions) with focus on promotion effectiveness, store segmentation, and inventory optimization.

## 🎯 Business Problem
Retailers face two critical challenges:
- **Stockouts**: Lost sales when demand exceeds inventory
- **Excess Inventory**: Wasted capital and storage costs

This model predicts daily unit sales at store-item level with **41% improvement** over baseline, enabling optimal inventory management and dynamic pricing decisions.

## 🏆 Key Achievements

| Metric | Value |
|--------|-------|
| **Final Holdout RMSLE** | **0.5189** (with store tiers + promotion buffer) |
| **Improvement vs Baseline** | **41.3%** |
| **Data Processed** | 10.4M rows (6-month window) |
| **Features Engineered** | 30+ features |
| **Stores Analyzed** | 54 stores (A/B/C tiered) |
| **Items Tracked** | 4,200+ items |

## 📈 Model Performance Summary

### Baseline Comparison
| Model | Validation RMSLE | Holdout RMSLE | Improvement vs Baseline |
|-------|-----------------|---------------|------------------------|
| Median Baseline | 0.8833 | 0.8833 | Reference |
| Mean Baseline | 1.0054 | - | -13.8% |
| SGD Lasso | 2.6122 | - | -195.7% |
| **Random Forest** | **0.4995** | **0.5721** | **35.2%** |
| **XGBoost** | **0.4982** | **0.5386** | **39.0%** |
| **XGBoost + Promotion Buffer** | **0.4982** | **0.5374** | **39.2%** |
| **XGBoost + Store Tiers + Buffer** | **0.4989** | **0.5189** | **41.3%** |

### Key Findings
- **Best Model**: XGBoost with store tiering + 30% promotion buffer
- **Validation to Holdout Drop**: Only 4.2% (excellent generalization)
- **Beats baseline by 41.3%** → Clear business value for deployment

## 🔧 Features Engineered (30+ Features)

### Time-Based Features
- `day_of_week` - Day of week (0-6)
- `day_of_month` - Day of month (1-31)
- `week_of_year` - Week number (1-52)
- `month` - Month (1-12)
- `quarter` - Quarter (1-4)
- `is_weekend` - Weekend flag
- `is_month_start/end` - Month boundary flags
- `is_quarter_end` - Quarter end flag
- `is_year_end` - Year end flag

### Lag Features (Historical Sales)
| Feature | Description | Null % |
|---------|-------------|-------|
| `sales_lag_1` | Yesterday's sales | 1.70% |
| `sales_lag_7` | Sales from 7 days ago | 11.71% |
| `sales_lag_14` | Sales from 14 days ago | 22.96% |
| `sales_lag_28` | Sales from 28 days ago | 43.89% |

### Rolling Statistics
| Feature | Description |
|---------|-------------|
| `sales_roll_mean_7/14/28` | Moving average (7,14,28 days) |
| `sales_roll_std_7/14/28` | Rolling volatility |

### Promotion Features
- `promo_lag_1` - Promotion status 1 day ago
- `promo_lag_7` - Promotion status 7 days ago
- `days_since_last_promo` - Days since last promotion

### Zero-Sale Patterns
- `is_zero_sale` - Zero-sale indicator
- `is_zero_lag_1` - Zero-sale yesterday
- `zero_count_last_7/28` - Zero-sale count in window
- `zero_ratio_last_7/28` - Zero-sale percentage in window

### Store Tiering (New!)
| Tier | Stores | Description |
|------|--------|-------------|
| Tier A | 12 stores | High volume (top 25%) |
| Tier B | 24 stores | Medium volume (middle 50%) |
| Tier C | 12 stores | Low volume (bottom 25%) |

### Outlier Detection
- `is_outlier_lag_1` - Sales > 3x rolling mean (2.28% of data flagged)

## 🧠 Model Explainability (SHAP Analysis)

### Top 10 Feature Importance
| Rank | Feature | Importance | Impact |
|------|---------|------------|--------|
| 1 | `sales_roll_mean_7` | 78.22% | Positive |
| 2 | `sales_lag_1` | 4.24% | Positive |
| 3 | `week_of_year` | 3.69% | Mixed |
| 4 | `sales_roll_mean_14` | 2.83% | Positive |
| 5 | `sales_lag_7` | 2.18% | Positive |
| 6 | `id` | 1.67% | Neutral |
| 7 | `day_of_week` | 1.59% | Mixed |
| 8 | `store_tier_A` | 1.38% | Positive |
| 9 | `onpromotion` | 1.32% | Positive |
| 10 | `month` | 1.29% | Mixed |

**Key Insight**: The 7-day rolling mean alone explains 78% of model decisions!

## 📊 Error Analysis

### By Store Tier
| Tier | MAE (units) | Under-prediction | Over-prediction |
|------|-------------|------------------|-----------------|
| Tier A (High Volume) | 6.25 | 48.4% | 51.6% |
| Tier B (Medium) | ~4.20 | Balanced | Balanced |
| Tier C (Low Volume) | 2.71 | Balanced | Balanced |

**Insight**: High-volume stores have 2.3x higher error rate

### By Promotion Status
| Condition | MAE (units) | Before Buffer | After 30% Buffer |
|-----------|-------------|---------------|------------------|
| Non-promotion | 4.06 | - | - |
| Promotion | 8.62 | 8.62 | 7.56 |

**Insight**: Promotion buffer reduced error by **12.3%** on promotion days

### By Day of Week
| Day | MAE (units) | Risk Level |
|-----|-------------|------------|
| Monday | ~4.5 | Low |
| Tuesday | ~4.0 | Low |
| Wednesday | 3.46 | **Best** |
| Thursday | ~4.0 | Low |
| Friday | ~4.5 | Low |
| Saturday | 5.07 | **Worst** |
| Sunday | ~5.0 | High |

**Recommendation**: Keep 20% safety stock on weekends

## 💼 Business Impact

### Inventory Optimization
- **Average daily sales**: 8 units per store-item
- **Prediction error**: 4.21 units MAE (52% relative error)
- **Stockout risk days**: 48.4% (balanced)
- **Excess inventory risk**: 51.6% (balanced)

### Promotion Strategy
- Promotions increase sales by 30-40% on average
- Model originally under-predicted promotions by 12%
- **Solution**: Applied 30% promotion buffer → 12.3% error reduction

### Store Segmentation
- **Tier A stores**: Need separate, more complex models
- **Tier C stores**: Baseline models may suffice (low volume)
- **Recommendation**: Build store-tier specific models for production

## 📁 Project Structure

demand-dynamic-pricing-ml-system/
├── main.py # Main execution script
├── notebooks/ # Jupyter notebooks with analysis
│ ├── 01_data_loading.ipynb
│ └── model_decisions.png # SHAP visualization
├── .gitignore # Git ignore rules (excludes 5GB train.csv)
└── README.md # This file


## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+

nstallation
bash
# Clone repository
git clone https://github.com/hari54haran98/demand-dynamic-pricing-ml-system.git
cd demand-dynamic-pricing-ml-system

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap



Data Setup
Download the dataset (train.csv - 5GB)

Place it in the project root folder

Run the analysis:

bash
python main.py

📊 Key Visualizations Generated
model_decisions.png - SHAP feature importance (top 10 features)

Daily sales trends and seasonality patterns

Promotion lift analysis

Store-level and item-level demand distributions

Error analysis by store tier, promotion status, and day of week

🔮 Future Improvements
Add external features (holidays, weather, economic indicators)

Implement deep learning (LSTM, Transformer) for time series

Build separate models for promotion vs non-promotion days

Create store-tier specific models (A/B/C)

Deploy as real-time API for dynamic pricing

Add A/B testing framework for pricing strategies

🛠️ Tech Stack
Python 3.8+ - Core language

pandas, numpy - Data processing (10.4M rows)

scikit-learn - Baseline models and metrics

XGBoost - Primary ML model (best performer)

SHAP - Model explainability

matplotlib, seaborn - Visualization

📝 Key Takeaways
What Worked Well ✅
Rolling mean features - 78% importance, excellent predictor

Store tiering - 41.3% improvement vs baseline

Promotion buffer - 12.3% error reduction on promotion days

Lag features - 1,7,14,28 day lags captured seasonality

Challenges & Solutions 🔧
Large data (10.4M rows) → Memory optimization with float32

Promotion under-prediction → 30% business rule buffer

Store heterogeneity → A/B/C tiering system

Weekend spikes → Day-specific error analysis

Production Readiness 🚀
✅ Model generalizes well (4.2% validation to holdout drop)

✅ Beats baseline by 41.3% → Clear business value

✅ Balanced predictions (48.4% under, 51.6% over)

✅ SHAP explainability for stakeholder trust

⚠️ Monitor promotion days and high-volume stores

👤 Author
GitHub: @hari54haran98

📄 License
MIT License - Free for academic and commercial use

For questions, issues, or suggestions, please open a GitHub issue!

text

---

## **Step 2: Add and Push README**

Run these commands:
```bash
git add README.md
git commit -m "Add comprehensive README with model results and analysis"
git push
Step 3: Verify on GitHub
Go to: https://github.com/hari54haran98/demand-dynamic-pricing-ml-system

You'll now see:

✅ Professional README with all your results

✅ Model performance tables

✅ Feature importance

✅ Error analysis

✅ Business insights














