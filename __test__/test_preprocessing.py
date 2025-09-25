#!/usr/bin/env python3
"""
Test script to demonstrate the preprocessing pipeline results
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def main():
    print("=== RAIN DATA PREPROCESSING DEMO ===\n")
    
    # Load original data
    df = pd.read_csv('../data/raw-rain-data.csv')
    print(f"📊 Original dataset: {df.shape}")
    print(f"🏛️  Unique provinces: {df['PROV_T'].nunique()}")
    print(f"📅 Year range: {df['YEAR'].min()} to {df['YEAR'].max()}")
    print(f"🌧️  Average rainfall range: {df['AvgRain'].min():.1f} to {df['AvgRain'].max():.1f}")
    
    # Start preprocessing
    df_processed = df.copy()
    
    # 1. Feature Engineering
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Fall'
    
    df_processed['Season'] = df_processed['MONTH'].apply(get_season)
    df_processed['RainRange'] = df_processed['MaxRain'] - df_processed['MinRain']
    df_processed['Quarter'] = ((df_processed['MONTH'] - 1) // 3) + 1
    df_processed['IsRainySeason'] = df_processed['MONTH'].apply(lambda x: 1 if x in [5,6,7,8,9,10] else 0)
    
    print(f"\n🔧 Feature Engineering:")
    print(f"   ✅ Season categories: {df_processed['Season'].unique()}")
    print(f"   ✅ Rainy season distribution: {df_processed['IsRainySeason'].value_counts().to_dict()}")
    
    # 2. One-Hot Encoding
    original_cols = len(df_processed.columns)
    
    # Province encoding
    province_encoded = pd.get_dummies(df_processed['PROV_T'], prefix='Province', drop_first=True)
    season_encoded = pd.get_dummies(df_processed['Season'], prefix='Season', drop_first=True)
    quarter_encoded = pd.get_dummies(df_processed['Quarter'], prefix='Q', drop_first=True)
    
    df_processed = pd.concat([df_processed, province_encoded, season_encoded, quarter_encoded], axis=1)
    df_processed = df_processed.drop(['PROV_T', 'Season', 'Quarter'], axis=1)
    
    print(f"\n🎯 One-Hot Encoding Results:")
    print(f"   📈 Shape before: ({df.shape[0]}, {original_cols})")
    print(f"   📈 Shape after: {df_processed.shape}")
    print(f"   🏛️  Province features: {province_encoded.shape[1]}")
    print(f"   🌱 Season features: {season_encoded.shape[1]}")
    print(f"   📅 Quarter features: {quarter_encoded.shape[1]}")
    
    # Show some of the new columns
    categorical_cols = [col for col in df_processed.columns if any(prefix in col for prefix in ['Province_', 'Season_', 'Q_'])]
    print(f"   🔢 Total categorical features: {len(categorical_cols)}")
    print(f"   📝 Sample features: {categorical_cols[:5]}")
    
    # 3. Feature Scaling
    numerical_features = ['MinRain', 'MaxRain', 'AvgRain']  # RainRange will be created after
    original_stats = df[numerical_features].describe()
    
    # Add RainRange to numerical features after it's created
    numerical_features.append('RainRange')
    scaler = MinMaxScaler()
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    
    scaled_stats = df_processed[numerical_features].describe()
    
    print(f"\n⚖️  Feature Scaling (MinMax 0-1):")
    print(f"   📊 Before scaling - AvgRain range: {original_stats.loc['min', 'AvgRain']:.1f} to {original_stats.loc['max', 'AvgRain']:.1f}")
    print(f"   📊 After scaling - AvgRain range: {scaled_stats.loc['min', 'AvgRain']:.3f} to {scaled_stats.loc['max', 'AvgRain']:.3f}")
    
    # 4. Final Summary
    feature_cols = [col for col in df_processed.columns if col not in ['AvgRain', 'YEAR', 'MONTH', 'PROV_ID']]
    
    print(f"\n🎉 PREPROCESSING COMPLETE!")
    print(f"   📋 Final dataset shape: {df_processed.shape}")
    print(f"   🎯 Ready for ML with {len(feature_cols)} features")
    print(f"   💾 Memory usage: {df_processed.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Save processed data
    df_processed.to_csv('data/processed_rain_data_demo.csv', index=False)
    print(f"   💾 Saved to: data/processed_rain_data_demo.csv")
    
    return df_processed

if __name__ == "__main__":
    main()
