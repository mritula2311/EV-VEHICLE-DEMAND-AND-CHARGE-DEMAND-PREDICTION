#!/usr/bin/env python3
"""
EV Vehicle Demand Prediction - Model Deployment and Prediction Pipeline
======================================================================

This script demonstrates how to deploy the trained models and create
a prediction pipeline for new data.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')


class EVDemandDeployment:
    """
    Deployment class for EV demand prediction models
    """
    
    def __init__(self, model_prefix='ev_demand_model'):
        """Initialize the deployment class"""
        self.model_prefix = model_prefix
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        
    def load_models(self):
        """Load the saved models and preprocessing components"""
        print("Loading trained models and preprocessing components...")
        
        try:
            # Load the best model
            self.model = joblib.load(f'{self.model_prefix}_best.joblib')
            print("✓ Model loaded successfully")
            
            # Load preprocessing components
            self.scaler = joblib.load(f'{self.model_prefix}_scaler.joblib')
            print("✓ Scaler loaded successfully")
            
            self.label_encoders = joblib.load(f'{self.model_prefix}_label_encoders.joblib')
            print("✓ Label encoders loaded successfully")
            
            # Load feature names
            self.feature_names = joblib.load(f'{self.model_prefix}_features.joblib')
            print("✓ Feature names loaded successfully")
            
            print(f"Model type: {type(self.model).__name__}")
            print(f"Expected features: {len(self.feature_names)}")
            
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            print("Please run the main training script first to generate the models.")
            raise
    
    def prepare_features_for_prediction(self, data):
        """
        Prepare features for prediction by applying the same transformations
        used during training
        """
        print("Preparing features for prediction...")
        
        # Make a copy to avoid modifying the original data
        processed_data = data.copy()
        
        # Convert date if it's a string
        if 'Date' in processed_data.columns:
            processed_data['Date'] = pd.to_datetime(processed_data['Date'])
        
        # Sort by County, State, and Date for proper feature engineering
        if all(col in processed_data.columns for col in ['County', 'State', 'Date']):
            processed_data = processed_data.sort_values(['County', 'State', 'Date'])
        
        # Create lag features (if there's enough historical data)
        if 'Electric Vehicle (EV) Total' in processed_data.columns:
            for lag in [1, 3, 6, 12]:
                processed_data[f'EV_Total_lag_{lag}'] = processed_data.groupby(['County', 'State'])['Electric Vehicle (EV) Total'].shift(lag)
        
        # Create rolling averages
        if 'Electric Vehicle (EV) Total' in processed_data.columns:
            for window in [3, 6, 12]:
                processed_data[f'EV_Total_rolling_{window}'] = processed_data.groupby(['County', 'State'])['Electric Vehicle (EV) Total'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
        
        # Create growth rate features
        if 'Electric Vehicle (EV) Total' in processed_data.columns:
            processed_data['EV_Total_pct_change'] = processed_data.groupby(['County', 'State'])['Electric Vehicle (EV) Total'].pct_change()
        
        # Create seasonal features
        if 'Month' in processed_data.columns:
            processed_data['Month_sin'] = np.sin(2 * np.pi * processed_data['Month'] / 12)
            processed_data['Month_cos'] = np.cos(2 * np.pi * processed_data['Month'] / 12)
        
        # Create interaction features
        if all(col in processed_data.columns for col in ['Battery Electric Vehicles (BEVs)', 'Plug-In Hybrid Electric Vehicles (PHEVs)']):
            processed_data['BEV_PHEV_ratio'] = processed_data['Battery Electric Vehicles (BEVs)'] / (
                processed_data['Plug-In Hybrid Electric Vehicles (PHEVs)'] + 1)
        
        # Handle categorical variables
        for col, encoder in self.label_encoders.items():
            if col in processed_data.columns:
                processed_data[col] = encoder.transform(processed_data[col].astype(str))
        
        # Fill missing values
        processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
        processed_data = processed_data.fillna(0)
        
        # Select only the features used in training
        feature_data = processed_data[self.feature_names]
        
        print(f"Features prepared. Shape: {feature_data.shape}")
        return feature_data
    
    def predict(self, data):
        """
        Make predictions on new data
        
        Args:
            data: DataFrame with the same structure as training data
            
        Returns:
            predictions: Array of predicted EV totals
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_models() first.")
        
        print("Making predictions...")
        
        # Prepare features
        X = self.prepare_features_for_prediction(data)
        
        # Handle scaling for Linear Regression
        if type(self.model).__name__ == 'LinearRegression':
            X = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        print(f"Predictions generated for {len(predictions)} samples")
        return predictions
    
    def predict_single_record(self, county, state, vehicle_use, bevs, phevs, 
                            non_ev_total, total_vehicles, percent_ev, 
                            year, month, day):
        """
        Make a prediction for a single record
        
        Args:
            county: County name
            state: State abbreviation
            vehicle_use: Vehicle primary use
            bevs: Battery Electric Vehicles count
            phevs: Plug-In Hybrid Electric Vehicles count
            non_ev_total: Non-Electric Vehicle Total
            total_vehicles: Total Vehicles
            percent_ev: Percent Electric Vehicles
            year: Year
            month: Month
            day: Day
            
        Returns:
            prediction: Predicted EV total
        """
        # Create a single record DataFrame
        record = pd.DataFrame({
            'County': [county],
            'State': [state],
            'Vehicle Primary Use': [vehicle_use],
            'Battery Electric Vehicles (BEVs)': [bevs],
            'Plug-In Hybrid Electric Vehicles (PHEVs)': [phevs],
            'Electric Vehicle (EV) Total': [bevs + phevs],  # For feature engineering
            'Non-Electric Vehicle Total': [non_ev_total],
            'Total Vehicles': [total_vehicles],
            'Percent Electric Vehicles': [percent_ev],
            'Year': [year],
            'Month': [month],
            'Day': [day],
            'Date': [pd.Timestamp(year, month, day)]
        })
        
        prediction = self.predict(record)
        return prediction[0]
    
    def batch_predict(self, input_file, output_file):
        """
        Make predictions on a batch of data from CSV file
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file with predictions
        """
        print(f"Processing batch predictions from {input_file}...")
        
        # Load data
        data = pd.read_csv(input_file)
        print(f"Loaded {len(data)} records for prediction")
        
        # Make predictions
        predictions = self.predict(data)
        
        # Add predictions to the data
        data['Predicted_EV_Total'] = predictions
        
        # Save results
        data.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        return data
    
    def create_sample_data(self, filename='sample_prediction_data.csv'):
        """Create sample data for testing predictions"""
        print(f"Creating sample data: {filename}")
        
        sample_data = pd.DataFrame({
            'County': ['Riverside', 'Prince William', 'Dakota', 'Ferry', 'Douglas'],
            'State': ['CA', 'VA', 'MN', 'WA', 'CO'],
            'Vehicle Primary Use': ['Passenger', 'Passenger', 'Passenger', 'Truck', 'Passenger'],
            'Battery Electric Vehicles (BEVs)': [10, 5, 2, 1, 3],
            'Plug-In Hybrid Electric Vehicles (PHEVs)': [5, 8, 3, 2, 4],
            'Electric Vehicle (EV) Total': [15, 13, 5, 3, 7],
            'Non-Electric Vehicle Total': [500, 200, 40, 3600, 90],
            'Total Vehicles': [515, 213, 45, 3603, 97],
            'Percent Electric Vehicles': [2.91, 6.10, 11.11, 0.08, 7.22],
            'Year': [2024, 2024, 2024, 2024, 2024],
            'Month': [6, 6, 6, 6, 6],
            'Day': [30, 30, 30, 30, 30],
            'Date': ['2024-06-30', '2024-06-30', '2024-06-30', '2024-06-30', '2024-06-30']
        })
        
        sample_data.to_csv(filename, index=False)
        print(f"Sample data saved to {filename}")
        return sample_data
    
    def generate_prediction_report(self, data, predictions):
        """Generate a prediction report"""
        print("\n" + "="*60)
        print("EV DEMAND PREDICTION REPORT")
        print("="*60)
        
        print(f"\nPREDICTION SUMMARY:")
        print(f"- Total records processed: {len(predictions)}")
        print(f"- Mean predicted EV total: {np.mean(predictions):.2f}")
        print(f"- Std predicted EV total: {np.std(predictions):.2f}")
        print(f"- Min predicted EV total: {np.min(predictions):.2f}")
        print(f"- Max predicted EV total: {np.max(predictions):.2f}")
        
        if 'State' in data.columns:
            print(f"\nPREDICTIONS BY STATE:")
            state_summary = data.groupby('State')['Predicted_EV_Total'].agg(['count', 'mean', 'sum']).round(2)
            print(state_summary.to_string())
        
        if 'Vehicle Primary Use' in data.columns:
            print(f"\nPREDICTIONS BY VEHICLE TYPE:")
            vehicle_summary = data.groupby('Vehicle Primary Use')['Predicted_EV_Total'].agg(['count', 'mean', 'sum']).round(2)
            print(vehicle_summary.to_string())
        
        print("\n" + "="*60)
    
    def demonstrate_deployment(self):
        """Demonstrate the complete deployment pipeline"""
        print("="*60)
        print("EV DEMAND PREDICTION - DEPLOYMENT DEMONSTRATION")
        print("="*60)
        
        # 1. Load models
        self.load_models()
        
        # 2. Create sample data
        sample_data = self.create_sample_data()
        
        # 3. Make batch predictions
        predictions = self.predict(sample_data)
        sample_data['Predicted_EV_Total'] = predictions
        
        # 4. Generate report
        self.generate_prediction_report(sample_data, predictions)
        
        # 5. Demonstrate single prediction
        print(f"\nSINGLE PREDICTION EXAMPLE:")
        single_pred = self.predict_single_record(
            county='Los Angeles',
            state='CA',
            vehicle_use='Passenger',
            bevs=50,
            phevs=30,
            non_ev_total=1000,
            total_vehicles=1080,
            percent_ev=7.41,
            year=2024,
            month=7,
            day=15
        )
        print(f"Predicted EV total for Los Angeles, CA: {single_pred:.2f}")
        
        # 6. Save predictions
        sample_data.to_csv('sample_predictions.csv', index=False)
        print(f"\nPredictions saved to 'sample_predictions.csv'")
        
        print("\nDeployment demonstration complete!")
        
        return sample_data


def main():
    """Main function to demonstrate deployment"""
    deployment = EVDemandDeployment()
    result_data = deployment.demonstrate_deployment()
    
    return deployment, result_data


if __name__ == "__main__":
    deployment, result_data = main()