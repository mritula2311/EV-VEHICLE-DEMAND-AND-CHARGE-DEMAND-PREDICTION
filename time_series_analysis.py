#!/usr/bin/env python3
"""
EV Vehicle Demand Prediction - Time Series Analysis and Advanced Error Analysis
==============================================================================

This script provides additional time series models (ARIMA) and detailed error analysis
to complement the main model improvements script.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")


class TimeSeriesEVAnalysis:
    """
    Time series analysis for EV demand prediction including ARIMA models
    """
    
    def __init__(self, data_path='preprocessed_ev_datacharge.csv'):
        """Initialize the time series analyzer"""
        self.data_path = data_path
        self.data = None
        self.ts_data = None
        self.arima_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for time series analysis"""
        print("Loading data for time series analysis...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.dropna()
        
        # Aggregate EV totals by date for time series analysis
        self.ts_data = self.data.groupby('Date')['Electric Vehicle (EV) Total'].sum().reset_index()
        self.ts_data.set_index('Date', inplace=True)
        self.ts_data = self.ts_data.asfreq('M')  # Monthly frequency
        
        print(f"Time series data shape: {self.ts_data.shape}")
        print(f"Date range: {self.ts_data.index.min()} to {self.ts_data.index.max()}")
        
    def analyze_time_series_components(self):
        """Analyze time series components (trend, seasonality, residuals)"""
        print("Analyzing time series components...")
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(
            self.ts_data['Electric Vehicle (EV) Total'], 
            model='additive', 
            period=12  # Monthly data, annual seasonality
        )
        
        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Original series
        axes[0].plot(self.ts_data.index, self.ts_data['Electric Vehicle (EV) Total'], 
                    color='blue', linewidth=1.5)
        axes[0].set_title('Original Time Series - Total EV Count', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(decomposition.trend.index, decomposition.trend, 
                    color='red', linewidth=1.5)
        axes[1].set_title('Trend Component', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal, 
                    color='green', linewidth=1.5)
        axes[2].set_title('Seasonal Component', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        axes[3].plot(decomposition.resid.index, decomposition.resid, 
                    color='orange', linewidth=1.5)
        axes[3].set_title('Residual Component', fontsize=14, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('time_series_decomposition.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return decomposition
    
    def plot_autocorrelation(self):
        """Plot autocorrelation and partial autocorrelation functions"""
        print("Plotting autocorrelation functions...")
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # ACF plot
        plot_acf(self.ts_data['Electric Vehicle (EV) Total'].dropna(), 
                ax=axes[0], lags=24, title='Autocorrelation Function (ACF)')
        
        # PACF plot
        plot_pacf(self.ts_data['Electric Vehicle (EV) Total'].dropna(), 
                 ax=axes[1], lags=24, title='Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        plt.savefig('autocorrelation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def fit_arima_models(self):
        """Fit ARIMA models with different parameters"""
        print("Fitting ARIMA models...")
        
        # Prepare data
        ts_values = self.ts_data['Electric Vehicle (EV) Total'].dropna()
        
        # Split data for ARIMA
        train_size = int(len(ts_values) * 0.8)
        train_data = ts_values[:train_size]
        test_data = ts_values[train_size:]
        
        # Try different ARIMA configurations
        arima_configs = [
            (1, 1, 1),  # ARIMA(1,1,1)
            (2, 1, 1),  # ARIMA(2,1,1)
            (1, 1, 2),  # ARIMA(1,1,2)
            (2, 1, 2),  # ARIMA(2,1,2)
            (1, 0, 1),  # ARIMA(1,0,1)
            (0, 1, 1),  # ARIMA(0,1,1)
        ]
        
        for p, d, q in arima_configs:
            try:
                model_name = f"ARIMA({p},{d},{q})"
                print(f"Fitting {model_name}...")
                
                # Fit model
                model = ARIMA(train_data, order=(p, d, q))
                fitted_model = model.fit()
                
                # Make predictions
                forecast = fitted_model.forecast(steps=len(test_data))
                
                # Calculate metrics
                mae = mean_absolute_error(test_data, forecast)
                mse = mean_squared_error(test_data, forecast)
                rmse = np.sqrt(mse)
                r2 = r2_score(test_data, forecast)
                
                # Store results
                self.arima_results[model_name] = {
                    'model': fitted_model,
                    'forecast': forecast,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
                
                print(f"{model_name} - AIC: {fitted_model.aic:.2f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"Failed to fit {model_name}: {str(e)}")
        
        # Find best model based on AIC
        if self.arima_results:
            best_model_name = min(self.arima_results.keys(), 
                                key=lambda k: self.arima_results[k]['aic'])
            print(f"\nBest ARIMA model: {best_model_name}")
            
        return train_data, test_data
    
    def visualize_arima_predictions(self, train_data, test_data):
        """Visualize ARIMA model predictions"""
        if not self.arima_results:
            print("No ARIMA results to visualize")
            return
        
        print("Creating ARIMA prediction visualizations...")
        
        # Get best model
        best_model_name = min(self.arima_results.keys(), 
                            key=lambda k: self.arima_results[k]['aic'])
        best_forecast = self.arima_results[best_model_name]['forecast']
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Plot training data
        train_dates = self.ts_data.index[:len(train_data)]
        test_dates = self.ts_data.index[len(train_data):len(train_data)+len(test_data)]
        
        plt.plot(train_dates, train_data, label='Training Data', color='blue', linewidth=2)
        plt.plot(test_dates, test_data, label='Actual Test Data', color='green', linewidth=2)
        plt.plot(test_dates, best_forecast, label=f'{best_model_name} Forecast', 
                color='red', linewidth=2, linestyle='--')
        
        plt.title(f'ARIMA Model Predictions - {best_model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Total EV Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('arima_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def comprehensive_error_analysis(self, train_data, test_data):
        """Perform comprehensive error analysis"""
        print("Performing comprehensive error analysis...")
        
        if not self.arima_results:
            print("No ARIMA results for error analysis")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, results in self.arima_results.items():
            comparison_data.append({
                'Model': model_name,
                'MAE': results['mae'],
                'RMSE': results['rmse'],
                'R²': results['r2'],
                'AIC': results['aic'],
                'BIC': results['bic']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # MAE comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['MAE'], color='lightcoral')
        axes[0, 0].set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['RMSE'], color='lightblue')
        axes[0, 1].set_title('Root Mean Squared Error Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['R²'], color='lightgreen')
        axes[1, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # AIC comparison
        axes[1, 1].bar(comparison_df['Model'], comparison_df['AIC'], color='gold')
        axes[1, 1].set_title('AIC Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('AIC')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('arima_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def generate_time_series_report(self, comparison_df):
        """Generate comprehensive time series analysis report"""
        print("\n" + "="*80)
        print("EV DEMAND PREDICTION - TIME SERIES ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nTIME SERIES DATA SUMMARY:")
        print(f"- Total time points: {len(self.ts_data)}")
        print(f"- Date range: {self.ts_data.index.min().strftime('%Y-%m-%d')} to {self.ts_data.index.max().strftime('%Y-%m-%d')}")
        print(f"- Frequency: Monthly")
        print(f"- Mean EV count: {self.ts_data['Electric Vehicle (EV) Total'].mean():.2f}")
        print(f"- Std EV count: {self.ts_data['Electric Vehicle (EV) Total'].std():.2f}")
        
        if not comparison_df.empty:
            print(f"\nARIMA MODEL COMPARISON:")
            print("-" * 30)
            print(comparison_df.to_string(index=False, float_format='%.4f'))
            
            best_model = comparison_df.loc[comparison_df['AIC'].idxmin()]
            print(f"\nBEST ARIMA MODEL: {best_model['Model']}")
            print(f"- AIC: {best_model['AIC']:.2f}")
            print(f"- R² Score: {best_model['R²']:.4f}")
            print(f"- RMSE: {best_model['RMSE']:.2f}")
        
        print(f"\nTIME SERIES INSIGHTS:")
        print("-" * 22)
        print("• EV adoption shows strong upward trend over time")
        print("• Seasonal patterns may exist in EV registration data")
        print("• ARIMA models can complement regression approaches for forecasting")
        print("• Time series decomposition reveals underlying patterns")
        
        print(f"\nRECOMMENDATIONS:")
        print("-" * 17)
        print("• Combine ARIMA forecasts with regression model predictions")
        print("• Monitor seasonal patterns for strategic planning")
        print("• Use time series analysis for long-term trend forecasting")
        print("• Consider external factors affecting temporal patterns")
        
        print("\n" + "="*80)
    
    def run_complete_time_series_analysis(self):
        """Run the complete time series analysis pipeline"""
        print("Starting comprehensive time series analysis...")
        
        # 1. Load and prepare data
        self.load_and_prepare_data()
        
        # 2. Analyze time series components
        decomposition = self.analyze_time_series_components()
        
        # 3. Plot autocorrelation
        self.plot_autocorrelation()
        
        # 4. Fit ARIMA models
        train_data, test_data = self.fit_arima_models()
        
        # 5. Visualize predictions
        self.visualize_arima_predictions(train_data, test_data)
        
        # 6. Error analysis
        comparison_df = self.comprehensive_error_analysis(train_data, test_data)
        
        # 7. Generate report
        self.generate_time_series_report(comparison_df)
        
        print("\nTime series analysis complete!")
        print("Generated files:")
        print("- time_series_decomposition.png")
        print("- autocorrelation_plots.png") 
        print("- arima_predictions.png")
        print("- arima_error_analysis.png")
        
        return self.arima_results, comparison_df


def main():
    """Main function to run time series analysis"""
    analyzer = TimeSeriesEVAnalysis()
    arima_results, comparison_df = analyzer.run_complete_time_series_analysis()
    
    return analyzer, arima_results, comparison_df


if __name__ == "__main__":
    analyzer, arima_results, comparison_df = main()