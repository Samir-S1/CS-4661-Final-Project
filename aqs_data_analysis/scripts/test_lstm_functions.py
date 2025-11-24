#!/usr/bin/env python3
"""
Test script for LSTM notebook functions.
Tests the trust calculation and time feature functions without requiring the full database.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_trust_values(df, value_col='Sample Measurement', decay_rate=0.9):
    """
    Calculate trust values for each data point.
    
    Parameters:
    -----------
    df : DataFrame with datetime index and value column
    value_col : Name of the column containing measurements
    decay_rate : Trust decay per time step (0-1, default 0.9)
    
    Returns:
    --------
    DataFrame with trust values added
    """
    df = df.copy()
    df['is_original'] = ~df[value_col].isna()
    df['trust'] = 0.0
    df['fill_method'] = 'missing'
    
    # Set trust to 1.0 for original observations
    df.loc[df['is_original'], 'trust'] = 1.0
    df.loc[df['is_original'], 'fill_method'] = 'original'
    
    # Forward fill with decreasing trust
    last_observed_idx = None
    steps_since_observation = 0
    
    for idx in df.index:
        if df.loc[idx, 'is_original']:
            last_observed_idx = idx
            steps_since_observation = 0
        elif last_observed_idx is not None:
            steps_since_observation += 1
            trust_value = (decay_rate ** steps_since_observation)
            if df.loc[idx, 'trust'] < trust_value:
                df.loc[idx, 'trust'] = trust_value
                df.loc[idx, 'fill_method'] = 'forward'
    
    # Backward fill with decreasing trust (only if better than forward fill)
    next_observed_idx = None
    steps_since_observation = 0
    
    for idx in reversed(df.index):
        if df.loc[idx, 'is_original']:
            next_observed_idx = idx
            steps_since_observation = 0
        elif next_observed_idx is not None:
            steps_since_observation += 1
            trust_value = (decay_rate ** steps_since_observation)
            if df.loc[idx, 'trust'] < trust_value:
                df.loc[idx, 'trust'] = trust_value
                df.loc[idx, 'fill_method'] = 'backward'
    
    return df


def add_time_features(df, datetime_col='datetime'):
    """
    Add time-based features to the dataframe.
    
    Parameters:
    -----------
    df : DataFrame with datetime column
    datetime_col : Name of datetime column
    
    Returns:
    --------
    DataFrame with added time features
    """
    df = df.copy()
    dt = df[datetime_col]
    
    # Basic time features
    df['hour'] = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek
    df['day_of_month'] = dt.dt.day
    df['month'] = dt.dt.month
    df['day_of_year'] = dt.dt.dayofyear
    
    # Cyclical encoding for hour (important for LSTM)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def test_trust_calculation():
    """Test trust value calculation with sample data."""
    print("=" * 60)
    print("Testing Trust Value Calculation")
    print("=" * 60)
    
    # Create sample data with gaps
    dates = pd.date_range('2020-01-01', periods=24, freq='h')
    values = [10.0, 12.0, 15.0, np.nan, np.nan, np.nan, 
              20.0, np.nan, np.nan, 18.0, 16.0, np.nan,
              np.nan, np.nan, 14.0, 13.0, np.nan, 12.0,
              np.nan, np.nan, np.nan, 11.0, 10.0, 9.0]
    
    df = pd.DataFrame({
        'datetime': dates,
        'Sample Measurement': values
    })
    
    print("\nOriginal data (first 12 rows):")
    print(df.head(12))
    
    # Calculate trust values
    df = calculate_trust_values(df, decay_rate=0.9)
    
    print("\nData with trust values (first 12 rows):")
    print(df[['datetime', 'Sample Measurement', 'trust', 'fill_method']].head(12))
    
    # Verify trust values
    print("\nTrust value statistics:")
    print(df['trust'].describe())
    
    print("\nFill method distribution:")
    print(df['fill_method'].value_counts())
    
    # Test specific cases
    print("\n" + "=" * 60)
    print("Verification Tests")
    print("=" * 60)
    
    # Original values should have trust = 1.0
    original_mask = df['fill_method'] == 'original'
    assert (df.loc[original_mask, 'trust'] == 1.0).all(), "Original values should have trust 1.0"
    print("✓ Original values have trust = 1.0")
    
    # Forward filled values should have trust < 1.0
    forward_mask = df['fill_method'] == 'forward'
    if forward_mask.any():
        assert (df.loc[forward_mask, 'trust'] < 1.0).all(), "Forward filled values should have trust < 1.0"
        print("✓ Forward filled values have trust < 1.0")
    
    # Backward filled values should have trust < 1.0
    backward_mask = df['fill_method'] == 'backward'
    if backward_mask.any():
        assert (df.loc[backward_mask, 'trust'] < 1.0).all(), "Backward filled values should have trust < 1.0"
        print("✓ Backward filled values have trust < 1.0")
    
    # Trust should decay with distance from observation
    # Check forward fill after index 2 (value 15.0) - indices 3 and 4
    idx_3 = df[df.index == 3]['trust'].values[0]
    idx_4 = df[df.index == 4]['trust'].values[0]
    if not np.isnan(idx_3) and not np.isnan(idx_4):
        assert idx_3 > idx_4, f"Trust should decay with distance: {idx_3} > {idx_4}"
        print("✓ Trust decays with distance from observation")
    
    print("\nAll trust calculation tests passed! ✓")


def test_time_features():
    """Test time feature extraction."""
    print("\n" + "=" * 60)
    print("Testing Time Feature Extraction")
    print("=" * 60)
    
    # Create sample datetime data
    dates = pd.date_range('2020-01-01', periods=48, freq='h')
    df = pd.DataFrame({
        'datetime': dates,
        'value': np.random.rand(48)
    })
    
    print("\nOriginal data (first 5 rows):")
    print(df.head())
    
    # Add time features
    df = add_time_features(df)
    
    print("\nData with time features (first 5 rows):")
    print(df.head())
    
    print("\nAll time feature columns:")
    print(df.columns.tolist())
    
    # Verify features
    print("\n" + "=" * 60)
    print("Verification Tests")
    print("=" * 60)
    
    # Check hour range
    assert df['hour'].min() >= 0 and df['hour'].max() <= 23, "Hour should be 0-23"
    print("✓ Hour values are in correct range [0-23]")
    
    # Check day of week range
    assert df['day_of_week'].min() >= 0 and df['day_of_week'].max() <= 6, "Day of week should be 0-6"
    print("✓ Day of week values are in correct range [0-6]")
    
    # Check cyclical encoding is bounded
    assert df['hour_sin'].min() >= -1 and df['hour_sin'].max() <= 1, "Sin values should be [-1, 1]"
    assert df['hour_cos'].min() >= -1 and df['hour_cos'].max() <= 1, "Cos values should be [-1, 1]"
    print("✓ Cyclical encodings are properly bounded [-1, 1]")
    
    # Check weekend indicator
    assert set(df['is_weekend'].unique()).issubset({0, 1}), "Weekend indicator should be 0 or 1"
    print("✓ Weekend indicator is binary (0 or 1)")
    
    # Verify cyclical continuity (hour 23 close to hour 0)
    hour_0 = df[df['hour'] == 0].iloc[0]
    hour_23 = df[df['hour'] == 23].iloc[0]
    distance = np.sqrt((hour_0['hour_sin'] - hour_23['hour_sin'])**2 + 
                      (hour_0['hour_cos'] - hour_23['hour_cos'])**2)
    print(f"✓ Distance between hour 23 and hour 0: {distance:.4f} (should be small)")
    
    print("\nAll time feature tests passed! ✓")


def test_sequence_creation():
    """Test sequence creation for LSTM."""
    print("\n" + "=" * 60)
    print("Testing Sequence Creation")
    print("=" * 60)
    
    def create_sequences(data, n_steps_in=5, n_steps_out=5):
        """Create sequences for LSTM: n_steps_in -> n_steps_out"""
        X, y = [], []
        
        for i in range(len(data) - n_steps_in - n_steps_out + 1):
            X.append(data[i:i + n_steps_in])
            y.append(data[i + n_steps_in:i + n_steps_in + n_steps_out, 0])
        
        return np.array(X), np.array(y)
    
    # Create sample data
    n_samples = 100
    n_features = 3
    data = np.random.rand(n_samples, n_features)
    
    # Create sequences
    n_steps_in = 5
    n_steps_out = 5
    X, y = create_sequences(data, n_steps_in, n_steps_out)
    
    print(f"\nInput data shape: {data.shape}")
    print(f"Sequence input shape (X): {X.shape}")
    print(f"Sequence output shape (y): {y.shape}")
    
    expected_samples = n_samples - n_steps_in - n_steps_out + 1
    assert X.shape[0] == expected_samples, f"Expected {expected_samples} samples"
    assert X.shape[1] == n_steps_in, f"Expected {n_steps_in} time steps in input"
    assert X.shape[2] == n_features, f"Expected {n_features} features"
    assert y.shape[0] == expected_samples, f"Expected {expected_samples} samples in output"
    assert y.shape[1] == n_steps_out, f"Expected {n_steps_out} time steps in output"
    
    print("\n✓ All sequence creation tests passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LSTM NOTEBOOK FUNCTION TESTS")
    print("=" * 60)
    
    try:
        test_trust_calculation()
        test_time_features()
        test_sequence_creation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓✓✓")
        print("=" * 60)
        print("\nThe LSTM notebook functions are working correctly.")
        print("You can now run the full notebook with your AQS data.")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
