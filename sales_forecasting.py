import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import joblib

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Statistical models
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')

class SalesForecastingSystem:
    """
    Comprehensive sales forecasting system using multiple time series 
    and machine learning approaches.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.data = None
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.performance_metrics = {}
        self.best_model = None
        self.best_model_name = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different forecasting models"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Moving Average': None,  # Will be implemented separately
            'ARIMA': None,  # Will be implemented separately
            'Exponential Smoothing': None  # Will be implemented separately
        }
    
    def generate_sample_data(self, 
                           start_date: str = '2020-01-01',
                           end_date: str = '2024-12-31',
                           freq: str = 'D') -> pd.DataFrame:
        """Generate realistic sales data with trends and seasonality"""
        
        np.random.seed(self.random_state)
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_periods = len(date_range)
        
        # Base trend (increasing over time)
        trend = np.linspace(1000, 2000, n_periods)
        
        # Seasonal patterns
        # Weekly seasonality (higher sales on weekends)
        weekly_pattern = 100 * np.sin(2 * np.pi * np.arange(n_periods) / 7)
        
        # Monthly seasonality (higher sales at month-end)
        monthly_pattern = 50 * np.sin(2 * np.pi * np.arange(n_periods) / 30)
        
        # Yearly seasonality (holiday seasons)
        yearly_pattern = 200 * np.sin(2 * np.pi * np.arange(n_periods) / 365.25)
        
        # Holiday effects (simplified)
        holiday_boost = np.zeros(n_periods)
        dates_df = pd.DataFrame({'date': date_range})
        
        # Black Friday boost (last Friday of November)
        for year in range(2020, 2025):
            black_friday = pd.Timestamp(f'{year}-11-01')
            # Find last Friday of November
            last_day = pd.Timestamp(f'{year}-11-30')
            while last_day.dayofweek != 4:  # Friday is 4
                last_day -= pd.Timedelta(days=1)
            
            if last_day in date_range:
                idx = date_range.get_loc(last_day)
                holiday_boost[max(0, idx-1):min(n_periods, idx+3)] += 500
        
        # Christmas season boost (December)
        for i, date in enumerate(date_range):
            if date.month == 12:
                holiday_boost[i] += 150 * (date.day / 31)
        
        # Economic impact (simulate COVID-19 impact)
        economic_impact = np.ones(n_periods)
        covid_start = pd.Timestamp('2020-03-15')
        covid_recovery = pd.Timestamp('2021-06-01')
        
        for i, date in enumerate(date_range):
            if covid_start <= date <= covid_recovery:
                # Gradual recovery
                days_since_start = (date - covid_start).days
                recovery_factor = min(1.0, days_since_start / 365)
                economic_impact[i] = 0.6 + 0.4 * recovery_factor
        
        # Random noise
        noise = np.random.normal(0, 50, n_periods)
        
        # Combine all components
        sales = (trend + 
                weekly_pattern + 
                monthly_pattern + 
                yearly_pattern + 
                holiday_boost) * economic_impact + noise
        
        # Ensure no negative sales
        sales = np.maximum(sales, 50)
        
        # Create additional features
        df = pd.DataFrame({
            'date': date_range,
            'sales': sales
        })
        
        # Add time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['week'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_month_end'] = (df['date'].dt.day >= 28).astype(int)
        df['is_quarter_end'] = df['date'].dt.month.isin([3, 6, 9, 12]).astype(int)
        
        # Holiday indicators
        df['is_december'] = (df['month'] == 12).astype(int)
        df['is_black_friday_week'] = 0
        
        # Mark Black Friday weeks
        for year in range(2020, 2025):
            black_friday = pd.Timestamp(f'{year}-11-01')
            last_day = pd.Timestamp(f'{year}-11-30')
            while last_day.dayofweek != 4:
                last_day -= pd.Timedelta(days=1)
            
            if last_day in date_range:
                week_start = last_day - pd.Timedelta(days=3)
                week_end = last_day + pd.Timedelta(days=3)
                mask = (df['date'] >= week_start) & (df['date'] <= week_end)
                df.loc[mask, 'is_black_friday_week'] = 1
        
        # Lagged features
        df['sales_lag_1'] = df['sales'].shift(1)
        df['sales_lag_7'] = df['sales'].shift(7)
        df['sales_lag_30'] = df['sales'].shift(30)
        df['sales_ma_7'] = df['sales'].rolling(window=7).mean()
        df['sales_ma_30'] = df['sales'].rolling(window=30).mean()
        
        # Drop rows with NaN values from lagged features
        df = df.dropna()
        
        return df
    
    def load_data(self, filepath: str = None, df: pd.DataFrame = None) -> pd.DataFrame:
        """Load sales data from file or DataFrame"""
        if df is not None:
            self.data = df
        elif filepath:
            self.data = pd.read_csv(filepath, parse_dates=['date'])
        else:
            print("Generating sample sales data...")
            self.data = self.generate_sample_data()
        
        # Ensure date column is datetime
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)
        
        return self.data
    
    def explore_data(self, target_column: str = 'sales'):
        """Perform exploratory data analysis"""
        
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        print("=== SALES DATA EXPLORATION ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"\nSales statistics:")
        print(self.data[target_column].describe())
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing values:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found.")
        
        # Create comprehensive visualizations
        self._create_exploration_plots(target_column)
    
    def _create_exploration_plots(self, target_column: str):
        """Create comprehensive EDA plots"""
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('Sales Data Exploration', fontsize=16, fontweight='bold')
        
        # Time series plot
        axes[0, 0].plot(self.data['date'], self.data[target_column], linewidth=0.8)
        axes[0, 0].set_title('Sales Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Sales')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sales distribution
        axes[0, 1].hist(self.data[target_column], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Sales Distribution')
        axes[0, 1].set_xlabel('Sales')
        axes[0, 1].set_ylabel('Frequency')
        
        # Monthly sales pattern
        monthly_sales = self.data.groupby('month')[target_column].mean()
        axes[1, 0].bar(monthly_sales.index, monthly_sales.values, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Average Sales by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Sales')
        axes[1, 0].set_xticks(range(1, 13))
        
        # Day of week pattern
        dow_sales = self.data.groupby('dayofweek')[target_column].mean()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 1].bar(range(7), dow_sales.values, alpha=0.7, color='orange')
        axes[1, 1].set_title('Average Sales by Day of Week')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Average Sales')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(dow_names)
        
        # Sales trend (yearly)
        yearly_sales = self.data.groupby('year')[target_column].mean()
        axes[2, 0].plot(yearly_sales.index, yearly_sales.values, marker='o', linewidth=2, markersize=8)
        axes[2, 0].set_title('Average Sales by Year')
        axes[2, 0].set_xlabel('Year')
        axes[2, 0].set_ylabel('Average Sales')
        
        # Quarterly pattern
        quarterly_sales = self.data.groupby('quarter')[target_column].mean()
        axes[2, 1].bar(quarterly_sales.index, quarterly_sales.values, alpha=0.7, color='lightcoral')
        axes[2, 1].set_title('Average Sales by Quarter')
        axes[2, 1].set_xlabel('Quarter')
        axes[2, 1].set_ylabel('Average Sales')
        
        plt.tight_layout()
        plt.show()
        
        # Decomposition plot
        self._plot_decomposition(target_column)
        
        # Correlation heatmap
        self._plot_correlation_heatmap()
    
    def _plot_decomposition(self, target_column: str, period: int = 365):
        """Plot time series decomposition"""
        
        # Set date as index for decomposition
        ts_data = self.data.set_index('date')[target_column]
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(ts_data, model='additive', period=period)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle('Time Series Decomposition', fontsize=16, fontweight='bold')
        
        # Original series
        decomposition.observed.plot(ax=axes[0], title='Original Sales Data')
        axes[0].set_ylabel('Sales')
        
        # Trend
        decomposition.trend.plot(ax=axes[1], title='Trend', color='red')
        axes[1].set_ylabel('Trend')
        
        # Seasonal
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='green')
        axes[2].set_ylabel('Seasonal')
        
        # Residual
        decomposition.resid.plot(ax=axes[3], title='Residual', color='orange')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap for numeric features"""
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                   cmap='coolwarm', center=0, square=True, 
                   linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def prepare_ml_features(self, target_column: str = 'sales') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for machine learning models"""
        
        if self.data is None:
            raise ValueError("No data loaded.")
        
        # Select features for ML models
        feature_columns = [
            'year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter',
            'is_weekend', 'is_month_end', 'is_quarter_end', 'is_december',
            'is_black_friday_week', 'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
            'sales_ma_7', 'sales_ma_30'
        ]
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        X = self.data[available_features]
        y = self.data[target_column]
        
        return X, y
    
    def train_ml_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train machine learning models"""
        
        print("=== TRAINING MACHINE LEARNING MODELS ===")
        
        # Time series split (respects temporal order)
        n_samples = len(X)
        split_point = int(n_samples * (1 - test_size))
        
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['ml_scaler'] = scaler
        
        results = []
        
        # Train ML models
        ml_models = ['Linear Regression', 'Random Forest']
        
        for name in ml_models:
            print(f"\nTraining {name}...")
            
            model = self.models[name]
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Store results
            model_results = {
                'Model': name,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Train_R2': train_r2,
                'Test_R2': test_r2
            }
            
            results.append(model_results)
            self.performance_metrics[name] = model_results
            self.predictions[name] = {
                'train': train_pred,
                'test': test_pred,
                'train_index': X_train.index,
                'test_index': X_test.index
            }
            
            print(f"  Test RMSE: {test_rmse:.2f}")
            print(f"  Test R²: {test_r2:.4f}")
        
        return X_test, y_test, results
    
    def train_statistical_models(self, target_column: str = 'sales', test_size: float = 0.2):
        """Train statistical time series models"""
        
        print("\n=== TRAINING STATISTICAL MODELS ===")
        
        ts_data = self.data.set_index('date')[target_column]
        
        # Split data
        n_samples = len(ts_data)
        split_point = int(n_samples * (1 - test_size))
        
        train_data = ts_data.iloc[:split_point]
        test_data = ts_data.iloc[split_point:]
        
        print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
        print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
        
        # 1. Moving Average
        print("\nTraining Moving Average...")
        ma_window = 30
        ma_predictions = train_data.rolling(window=ma_window).mean()
        ma_forecast = [ma_predictions.iloc[-1]] * len(test_data)
        
        # Calculate metrics for MA
        ma_test_rmse = np.sqrt(mean_squared_error(test_data, ma_forecast))
        ma_test_mae = mean_absolute_error(test_data, ma_forecast)
        ma_test_r2 = r2_score(test_data, ma_forecast)
        
        self.performance_metrics['Moving Average'] = {
            'Model': 'Moving Average',
            'Test_RMSE': ma_test_rmse,
            'Test_MAE': ma_test_mae,
            'Test_R2': ma_test_r2
        }
        
        self.predictions['Moving Average'] = {
            'test': ma_forecast,
            'test_index': test_data.index
        }
        
        print(f"  Test RMSE: {ma_test_rmse:.2f}")
        print(f"  Test R²: {ma_test_r2:.4f}")
        
        # 2. ARIMA Model
        print("\nTraining ARIMA...")
        try:
            # Auto-determine ARIMA parameters (simplified approach)
            arima_model = ARIMA(train_data, order=(1, 1, 1))
            arima_fitted = arima_model.fit()
            
            # Forecast
            arima_forecast = arima_fitted.forecast(steps=len(test_data))
            
            # Calculate metrics for ARIMA
            arima_test_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
            arima_test_mae = mean_absolute_error(test_data, arima_forecast)
            arima_test_r2 = r2_score(test_data, arima_forecast)
            
            self.performance_metrics['ARIMA'] = {
                'Model': 'ARIMA',
                'Test_RMSE': arima_test_rmse,
                'Test_MAE': arima_test_mae,
                'Test_R2': arima_test_r2
            }
            
            self.predictions['ARIMA'] = {
                'test': arima_forecast.values,
                'test_index': test_data.index,
                'model': arima_fitted
            }
            
            print(f"  Test RMSE: {arima_test_rmse:.2f}")
            print(f"  Test R²: {arima_test_r2:.4f}")
            
        except Exception as e:
            print(f"  ARIMA training failed: {str(e)}")
        
        # 3. Exponential Smoothing
        print("\nTraining Exponential Smoothing...")
        try:
            # Holt-Winters Exponential Smoothing
            exp_model = ExponentialSmoothing(
                train_data, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=365
            )
            exp_fitted = exp_model.fit()
            
            # Forecast
            exp_forecast = exp_fitted.forecast(steps=len(test_data))
            
            # Calculate metrics for Exponential Smoothing
            exp_test_rmse = np.sqrt(mean_squared_error(test_data, exp_forecast))
            exp_test_mae = mean_absolute_error(test_data, exp_forecast)
            exp_test_r2 = r2_score(test_data, exp_forecast)
            
            self.performance_metrics['Exponential Smoothing'] = {
                'Model': 'Exponential Smoothing',
                'Test_RMSE': exp_test_rmse,
                'Test_MAE': exp_test_mae,
                'Test_R2': exp_test_r2
            }
            
            self.predictions['Exponential Smoothing'] = {
                'test': exp_forecast.values,
                'test_index': test_data.index,
                'model': exp_fitted
            }
            
            print(f"  Test RMSE: {exp_test_rmse:.2f}")
            print(f"  Test R²: {exp_test_r2:.4f}")
            
        except Exception as e:
            print(f"  Exponential Smoothing training failed: {str(e)}")
        
        return test_data
    
    def find_best_model(self):
        """Find the best performing model based on test RMSE"""
        
        if not self.performance_metrics:
            print("No models trained yet.")
            return
        
        best_rmse = float('inf')
        best_name = None
        
        for name, metrics in self.performance_metrics.items():
            if metrics['Test_RMSE'] < best_rmse:
                best_rmse = metrics['Test_RMSE']
                best_name = name
        
        self.best_model_name = best_name
        
        # Store the best model object
        if best_name in ['Linear Regression', 'Random Forest']:
            self.best_model = self.models[best_name]
        elif best_name in self.predictions:
            self.best_model = self.predictions[best_name].get('model')
        
        print(f"\n=== BEST MODEL: {best_name} ===")
        print(f"Test RMSE: {best_rmse:.2f}")
        print(f"Test R²: {self.performance_metrics[best_name]['Test_R2']:.4f}")
        
        return best_name
    
    def plot_model_comparison(self):
        """Plot comparison of all models"""
        
        if not self.performance_metrics:
            print("No model results available.")
            return
        
        # Create DataFrame from results
        results_df = pd.DataFrame(list(self.performance_metrics.values()))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # RMSE comparison
        axes[0].bar(results_df['Model'], results_df['Test_RMSE'], alpha=0.7, color='skyblue')
        axes[0].set_title('Test RMSE by Model')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1].bar(results_df['Model'], results_df['Test_MAE'], alpha=0.7, color='lightgreen')
        axes[1].set_title('Test MAE by Model')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[2].bar(results_df['Model'], results_df['Test_R2'], alpha=0.7, color='lightcoral')
        axes[2].set_title('Test R² by Model')
        axes[2].set_ylabel('R²')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, target_column: str = 'sales'):
        """Plot actual vs predicted values for all models"""
        
        if not self.predictions:
            print("No predictions available.")
            return
        
        # Get test data for comparison
        ts_data = self.data.set_index('date')[target_column]
        
        fig, axes = plt.subplots(len(self.predictions), 1, figsize=(15, 4*len(self.predictions)))
        if len(self.predictions) == 1:
            axes = [axes]
        
        fig.suptitle('Actual vs Predicted Sales', fontsize=16, fontweight='bold')
        
        for i, (model_name, pred_data) in enumerate(self.predictions.items()):
            ax = axes[i]
            
            # Plot actual values
            test_index = pred_data['test_index']
            actual_values = ts_data.loc[test_index]
            
            ax.plot(test_index, actual_values, label='Actual', linewidth=2, color='blue')
            ax.plot(test_index, pred_data['test'], label='Predicted', 
                   linewidth=2, color='red', linestyle='--')
            
            ax.set_title(f'{model_name} - Test Period Predictions')
            ax.set_xlabel('Date')
            ax.set_ylabel('Sales')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add RMSE to the plot
            rmse = self.performance_metrics[model_name]['Test_RMSE']
            ax.text(0.02, 0.95, f'RMSE: {rmse:.2f}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def forecast_future(self, periods: int = 30, model_name: str = None) -> pd.DataFrame:
        """Forecast future sales"""
        
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name is None:
            raise ValueError("No best model found. Train models first.")
        
        print(f"Generating {periods}-day forecast using {model_name}...")
        
        # Generate future dates
        last_date = self.data['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        if model_name in ['Linear Regression', 'Random Forest']:
            # ML models need features
            future_forecast = self._forecast_ml_model(future_dates, model_name)
        else:
            # Statistical models
            future_forecast = self._forecast_statistical_model(future_dates, model_name)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecasted_sales': future_forecast,
            'model': model_name
        })
        
        return forecast_df
    
    def _forecast_ml_model(self, future_dates: pd.DatetimeIndex, model_name: str) -> np.ndarray:
        """Generate forecasts using ML models"""
        
        # Create features for future dates
        future_features = []
        
        for date in future_dates:
            features = {
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'dayofweek': date.dayofweek,
                'dayofyear': date.dayofyear,
                'week': date.isocalendar().week,
                'quarter': date.quarter,
                'is_weekend': int(date.dayofweek >= 5),
                'is_month_end': int(date.day >= 28),
                'is_quarter_end': int(date.month in [3, 6, 9, 12]),
                'is_december': int(date.month == 12),
                'is_black_friday_week': 0  # Simplified
            }
            
            # For lagged features, use recent values (simplified approach)
            recent_sales = self.data['sales'].tail(30).mean()
            features.update({
                'sales_lag_1': recent_sales,
                'sales_lag_7': recent_sales,
                'sales_lag_30': recent_sales,
                'sales_ma_7': recent_sales,
                'sales_ma_30': recent_sales
            })
            
            future_features.append(features)
        
        # Convert to DataFrame
        future_df = pd.DataFrame(future_features)
        
        # Scale features
        scaler = self.scalers['ml_scaler']
        future_scaled = scaler.transform(future_df)
        
        # Make predictions
        model = self.models[model_name]
        forecast = model.predict(future_scaled)
        
        return forecast
    
    def _forecast_statistical_model(self, future_dates: pd.DatetimeIndex, model_name: str) -> np.ndarray:
        """Generate forecasts using statistical models"""
        
        if model_name == 'Moving Average':
            # Simple moving average forecast
            recent_avg = self.data['sales'].tail(30).mean()
            return np.full(len(future_dates), recent_avg)
        
        elif model_name in self.predictions and 'model' in self.predictions[model_name]:
            # Use fitted statistical model
            fitted_model = self.predictions[model_name]['model']
            forecast = fitted_model.forecast(steps=len(future_dates))
            return forecast.values if hasattr(forecast, 'values') else forecast
        
        else:
            raise ValueError(f"Cannot generate forecast for {model_name}")
    
    def plot_forecast(self, forecast_df: pd.DataFrame, target_column: str = 'sales'):
        """Plot historical data with future forecast"""
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # Plot historical data
        ax.plot(self.data['date'], self.data[target_column], 
               label='Historical Sales', linewidth=2, color='blue')
        
        # Plot forecast
        ax.plot(forecast_df['date'], forecast_df['forecasted_sales'], 
               label=f'Forecast ({forecast_df["model"].iloc[0]})', 
               linewidth=2, color='red', linestyle='--')
        
        # Add vertical line to separate historical and forecast
        last_historical_date = self.data['date'].max()
        ax.axvline(x=last_historical_date, color='gray', linestyle=':', alpha=0.7)
        
        ax.set_title('Sales Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_forecast_intervals(self, forecast_df: pd.DataFrame, 
                                   confidence_level: float = 0.95) -> pd.DataFrame:
        """Calculate confidence intervals for forecasts"""
        
        # Simple approach: use historical residuals to estimate uncertainty
        model_name = forecast_df['model'].iloc[0]
        
        if model_name in self.predictions:
            # Calculate residuals from test set
            pred_data = self.predictions[model_name]
            test_index = pred_data['test_index']
            actual = self.data.set_index('date')['sales'].loc[test_index]
            predicted = pred_data['test']
            residuals = actual - predicted
            
            # Calculate prediction interval
            residual_std = np.std(residuals)
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # 99%
            
            margin_of_error = z_score * residual_std
            
            forecast_df['lower_bound'] = forecast_df['forecasted_sales'] - margin_of_error
            forecast_df['upper_bound'] = forecast_df['forecasted_sales'] + margin_of_error
            
        return forecast_df
    
    def save_model(self, filepath: str):
        """Save the forecasting system"""
        
        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'performance_metrics': self.performance_metrics,
            'best_model_name': self.best_model_name,
            'predictions': {k: {key: val for key, val in v.items() 
                              if key != 'model'} for k, v in self.predictions.items()}
        }
        
        joblib.dump(save_data, filepath)
        print(f"Forecasting system saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved forecasting system"""
        
        save_data = joblib.load(filepath)
        
        self.models = save_data['models']
        self.scalers = save_data['scalers']
        self.performance_metrics = save_data['performance_metrics']
        self.best_model_name = save_data['best_model_name']
        self.predictions = save_data['predictions']
        
        print(f"Forecasting system loaded from {filepath}")
        print(f"Best model: {self.best_model_name}")

def main():
    """Main execution function"""
    
    print("=== SALES FORECASTING SYSTEM ===")
    
    # Initialize the forecasting system
    forecaster = SalesForecastingSystem(random_state=42)
    
    # Load or generate data
    data = forecaster.load_data()  # This will generate sample data
    print(f"Loaded {len(data)} days of sales data")
    
    # Explore the data
    forecaster.explore_data(target_column='sales')
    
    # Prepare features for ML models
    X, y = forecaster.prepare_ml_features(target_column='sales')
    
    # Train machine learning models
    X_test, y_test, ml_results = forecaster.train_ml_models(X, y, test_size=0.2)
    
    # Train statistical models
    test_data = forecaster.train_statistical_models(target_column='sales', test_size=0.2)
    
    # Find best model
    best_model = forecaster.find_best_model()
    
    # Plot model comparison
    forecaster.plot_model_comparison()
    
    # Plot predictions
    forecaster.plot_predictions(target_column='sales')
    
    # Generate future forecast
    future_forecast = forecaster.forecast_future(periods=60, model_name=best_model)
    
    # Add confidence intervals
    future_forecast = forecaster.calculate_forecast_intervals(future_forecast)
    
    # Plot forecast
    forecaster.plot_forecast(future_forecast, target_column='sales')
    
    print("\n=== FORECAST SUMMARY ===")
    print(f"Best model: {best_model}")
    print(f"60-day forecast generated")
    print(f"Average forecasted sales: {future_forecast['forecasted_sales'].mean():.2f}")
    print(f"Forecast range: {future_forecast['forecasted_sales'].min():.2f} - {future_forecast['forecasted_sales'].max():.2f}")
    
    # Save the forecasting system
    forecaster.save_model('sales_forecasting_system.pkl')
    
    # Display first few forecast values
    print("\nFirst 10 days forecast:")
    print(future_forecast[['date', 'forecasted_sales', 'lower_bound', 'upper_bound']].head(10))

if __name__ == "__main__":
    main()