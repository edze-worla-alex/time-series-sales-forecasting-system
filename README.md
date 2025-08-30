# Time Series Sales Forecasting System

A comprehensive sales forecasting system that combines multiple machine learning and statistical approaches to predict future sales. The system includes trend analysis, seasonality detection, and confidence interval estimation for robust business planning.

## Author
**Edze Worla Alex**

## Features

- **Multiple Forecasting Models**: Linear Regression, Random Forest, ARIMA, Exponential Smoothing, Moving Average
- **Time Series Analysis**: Seasonal decomposition, trend analysis, stationarity testing
- **Feature Engineering**: Automated creation of time-based features and lagged variables
- **Model Comparison**: Comprehensive evaluation using multiple metrics (RMSE, MAE, R²)
- **Future Forecasting**: Generate predictions with confidence intervals
- **Visualization Suite**: Comprehensive plotting for EDA and results analysis
- **Model Persistence**: Save and load trained forecasting systems

## Business Problem

Sales forecasting is critical for:
- **Inventory Management**: Optimize stock levels and reduce carrying costs
- **Resource Planning**: Allocate staff and resources efficiently
- **Financial Planning**: Budget preparation and revenue projections
- **Marketing Strategy**: Time promotional campaigns effectively
- **Supply Chain Optimization**: Coordinate with suppliers and distributors

## Technical Architecture

### Data Components
- **Temporal Features**: Year, month, day, week, quarter
- **Cyclical Patterns**: Day of week, seasonality indicators
- **Lagged Variables**: Previous sales values (1, 7, 30 days)
- **Moving Averages**: Short and long-term trend indicators
- **Event Indicators**: Holiday seasons, special events
- **External Factors**: Economic indicators, weather effects

### Model Categories

#### Machine Learning Models
1. **Linear Regression**: Baseline linear relationships
2. **Random Forest**: Non-linear patterns and feature interactions

#### Statistical Models
3. **Moving Average**: Simple trend-following approach
4. **ARIMA**: Autoregressive Integrated Moving Average
5. **Exponential Smoothing**: Holt-Winters triple exponential smoothing

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
statsmodels>=0.13.0
joblib>=1.0.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/edze-worla-alex/time-series-sales-forecasting-system.git
cd time-series-sales-forecasting-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from sales_forecasting import SalesForecastingSystem
import pandas as pd

# Initialize the forecasting system
forecaster = SalesForecastingSystem(random_state=42)

# Load your data (or generate sample data)
data = forecaster.load_data('your_sales_data.csv')  # or forecaster.load_data() for sample data

# Explore the data
forecaster.explore_data(target_column='sales')

# Prepare features and train models
X, y = forecaster.prepare_ml_features(target_column='sales')
X_test, y_test, ml_results = forecaster.train_ml_models(X, y, test_size=0.2)
test_data = forecaster.train_statistical_models(target_column='sales', test_size=0.2)

# Find best model and compare performance
best_model = forecaster.find_best_model()
forecaster.plot_model_comparison()

# Generate future forecast
future_forecast = forecaster.forecast_future(periods=30)
forecaster.plot_forecast(future_forecast)

print(f"Best model: {best_model}")
print(f"30-day forecast generated")
```

### Advanced Usage

```python
# Custom forecast with confidence intervals
future_forecast = forecaster.forecast_future(periods=60, model_name='Random Forest')
future_forecast = forecaster.calculate_forecast_intervals(future_forecast, confidence_level=0.95)

# Plot predictions vs actual
forecaster.plot_predictions(target_column='sales')

# Save the trained system
forecaster.save_model('my_forecasting_system.pkl')

# Load a saved system
new_forecaster = SalesForecastingSystem()
new_forecaster.load_model('my_forecasting_system.pkl')

# Generate forecasts with the loaded system
new_forecast = new_forecaster.forecast_future(periods=30)
```

## Data Format

Your CSV file should contain at least a date column and sales column:

```csv
date,sales
2020-01-01,1234.56
2020-01-02,1456.78
2020-01-03,1123.45
...
```

The system automatically generates additional features:
- Time-based features (year, month, day, etc.)
- Lagged sales values
- Moving averages
- Holiday and event indicators
- Weekend/weekday flags

## Model Performance Examples

Typical performance on sample data:

```
=== MODEL COMPARISON ===
Model                  Test_RMSE   Test_MAE   Test_R²
Random Forest          89.23       67.45      0.924
Linear Regression      134.56      98.76      0.856
Exponential Smoothing  156.78      112.34     0.798
ARIMA                  178.90      123.45     0.743
Moving Average         234.56      167.89     0.621

Best Model: Random Forest
Test RMSE: 89.23
Test R²: 0.924
```

## Feature Importance Analysis

The system automatically identifies key drivers:

1. **Historical Sales** (40-60% importance)
   - Recent sales values (lag 1, 7, 30 days)
   - Moving averages (7, 30 days)

2. **Seasonal Patterns** (20-30% importance)
   - Month of year
   - Day of week
   - Quarter

3. **Special Events** (10-20% importance)
   - Holiday seasons
   - End of month/quarter effects
   - Weekend indicators

## Comprehensive Visualizations

### 1. Exploratory Data Analysis
- **Time Series Plot**: Sales over time with trend identification
- **Distribution Analysis**: Sales value distribution and outliers
- **Seasonal Patterns**: Monthly, weekly, and quarterly patterns
- **Correlation Heatmap**: Relationships between features
- **Decomposition Plot**: Trend, seasonal, and residual components

### 2. Model Performance
- **Comparison Charts**: RMSE, MAE, and R² across all models
- **Prediction Plots**: Actual vs predicted for each model
- **Residual Analysis**: Error patterns and model diagnostics
- **Forecast Visualization**: Historical data with future predictions

### 3. Advanced Analytics
- **Confidence Intervals**: Uncertainty quantification
- **Feature Importance**: Driver analysis for business insights
- **Seasonal Decomposition**: Trend and cyclical component analysis

## Business Insights

### Seasonal Patterns
- **Weekly Cycles**: Typically higher sales on weekends
- **Monthly Patterns**: End-of-month purchasing behavior
- **Quarterly Effects**: Budget cycles and seasonal demand
- **Annual Trends**: Holiday seasons and economic cycles

### Key Performance Drivers
- **Historical Performance**: Strong predictor of future sales
- **External Events**: Holidays, promotions, economic factors
- **Market Conditions**: Seasonal demand patterns
- **Business Cycles**: Monthly/quarterly purchasing patterns

## API Reference

### SalesForecastingSystem Class

#### Core Methods

**`__init__(random_state=42)`**
- Initialize the forecasting system
- Set random seed for reproducible results

**`load_data(filepath=None, df=None)`**
- Load sales data from CSV or DataFrame
- Automatically handles date parsing and sorting

**`explore_data(target_column='sales')`**
- Comprehensive exploratory data analysis
- Generates visualization suite

**`prepare_ml_features(target_column='sales')`**
- Engineer features for machine learning models
- Creates lagged variables and time-based features

**`train_ml_models(X, y, test_size=0.2)`**
- Train machine learning models (Linear Regression, Random Forest)
- Uses time series split to maintain temporal order

**`train_statistical_models(target_column='sales', test_size=0.2)`**
- Train statistical models (ARIMA, Exponential Smoothing, Moving Average)
- Handles model-specific requirements automatically

**`find_best_model()`**
- Identify best performing model based on test RMSE
- Returns model name and performance metrics

**`forecast_future(periods=30, model_name=None)`**
- Generate future predictions
- Uses best model if none specified

**`calculate_forecast_intervals(forecast_df, confidence_level=0.95)`**
- Add confidence intervals to forecasts
- Based on historical residual analysis

#### Visualization Methods

**`plot_model_comparison()`**
- Compare all models on RMSE, MAE, and R²

**`plot_predictions(target_column='sales')`**
- Show actual vs predicted for all models

**`plot_forecast(forecast_df, target_column='sales')`**
- Visualize historical data with future forecast

#### Utility Methods

**`save_model(filepath)`**
- Save entire forecasting system

**`load_model(filepath)`**
- Load saved forecasting system

## File Structure

```
sales-forecasting-system/
├── sales_forecasting.py          # Main implementation
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── examples/
│   ├── basic_forecasting.py      # Basic usage example
│   ├── advanced_analysis.py      # Advanced features demo
│   └── business_dashboard.py     # Dashboard example
├── data/
│   ├── sample_sales_data.csv     # Sample dataset
│   └── data_dictionary.txt       # Feature descriptions
├── models/
│   └── pretrained_system.pkl     # Example saved system
├── notebooks/
│   ├── sales_analysis.ipynb      # Detailed analysis notebook
│   ├── model_comparison.ipynb    # Model evaluation notebook
│   └── forecasting_demo.ipynb    # Interactive demo
└── tests/
    ├── test_forecasting.py       # Unit tests
    └── test_models.py             # Model testing
```

## Business Applications

### Retail Operations
1. **Inventory Planning**: Optimize stock levels by product/location
2. **Staff Scheduling**: Align workforce with predicted demand
3. **Pricing Strategy**: Dynamic pricing based on demand forecasts
4. **Promotion Planning**: Time marketing campaigns effectively

### Supply Chain Management
1. **Procurement Planning**: Order materials based on sales forecasts
2. **Warehouse Optimization**: Allocate space efficiently
3. **Distribution Planning**: Optimize delivery schedules
4. **Vendor Management**: Share forecasts with suppliers

### Financial Planning
1. **Revenue Projections**: Accurate budget preparation
2. **Cash Flow Management**: Predict incoming revenue
3. **Investment Planning**: Capital allocation decisions
4. **Risk Assessment**: Identify potential shortfalls

### Strategic Planning
1. **Market Analysis**: Understand demand patterns
2. **Product Development**: Identify growth opportunities
3. **Expansion Planning**: Forecast demand in new markets
4. **Competitive Analysis**: Benchmark against market trends

## Model Selection Guidelines

### Choose Linear Regression when:
- Simple, interpretable relationships needed
- Limited historical data available
- Baseline model for comparison
- Fast training and prediction required

### Choose Random Forest when:
- Non-linear patterns in data
- Feature interactions important
- Robust performance needed
- Feature importance analysis required

### Choose ARIMA when:
- Strong temporal dependencies
- Stationary time series (after differencing)
- Classical statistical approach preferred
- Confidence intervals needed

### Choose Exponential Smoothing when:
- Clear trend and seasonal patterns
- Medium-term forecasting horizon
- Smooth, stable forecasts needed
- Limited computational resources

### Choose Moving Average when:
- Simple trend following needed
- Very short-term forecasting
- Baseline comparison model
- Real-time implementation required

## Performance Optimization

### For Large Datasets
```python
# Use data sampling for initial development
sample_data = data.sample(frac=0.1, random_state=42)

# Optimize Random Forest parameters
forecaster.models['Random Forest'].set_params(
    n_estimators=50,  # Reduce from default 100
    n_jobs=-1,        # Use all CPU cores
    max_depth=10      # Limit tree depth
)
```

### For Real-time Forecasting
```python
# Use simpler models for faster predictions
fast_forecaster = SalesForecastingSystem()
fast_forecaster.models = {
    'Linear Regression': LinearRegression(),
    'Moving Average': None
}
```

## Troubleshooting

### Common Issues and Solutions

**1. ARIMA Model Fails to Train**
```python
# Check data stationarity
from stats
