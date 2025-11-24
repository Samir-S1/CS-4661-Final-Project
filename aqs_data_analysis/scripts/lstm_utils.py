"""
Utility functions for LSTM-based PM2.5 prediction model.

This module contains shared functions used across the LSTM notebooks and test scripts.
"""

import pandas as pd
import numpy as np


def calculate_trust_values(df, value_col='Sample Measurement', decay_rate=0.9):
    """
    Calculate trust values for each data point based on distance from observations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index and value column
    value_col : str
        Name of the column containing measurements
    decay_rate : float
        Trust decay per time step (0-1, default 0.9)
        - 0.9 means 90% trust retention per hour
        - Higher values = slower decay
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with trust values and fill method added
        
    Notes:
    ------
    - Original observations get trust = 1.0
    - Forward filled values get trust = decay_rate^steps_from_observation
    - Backward filled values use the same calculation
    - When both forward and backward fills are possible, the higher trust value is used
    
    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'datetime': pd.date_range('2020-01-01', periods=5, freq='h'),
    ...     'Sample Measurement': [10.0, np.nan, np.nan, 15.0, np.nan]
    ... })
    >>> df = calculate_trust_values(df, decay_rate=0.9)
    >>> print(df['trust'].tolist())
    [1.0, 0.9, 0.81, 1.0, 0.9]
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
    Add time-based features to the dataframe for LSTM model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime column
    datetime_col : str
        Name of datetime column (default: 'datetime')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added time features
        
    Features Added:
    ---------------
    - hour: Hour of day (0-23)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - day_of_month: Day of month (1-31)
    - month: Month (1-12)
    - day_of_year: Day of year (1-366)
    - hour_sin, hour_cos: Cyclical encoding of hour
    - dow_sin, dow_cos: Cyclical encoding of day of week
    - month_sin, month_cos: Cyclical encoding of month
    - is_weekend: Binary indicator (1=weekend, 0=weekday)
    
    Notes:
    ------
    Cyclical encoding ensures the model understands temporal continuity:
    - Hour 23 is close to hour 0 (end of day wraps to start)
    - Sunday is close to Monday (week wraps)
    - December is close to January (year wraps)
    
    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'datetime': pd.date_range('2020-01-01', periods=24, freq='h')
    ... })
    >>> df = add_time_features(df)
    >>> print(df.columns.tolist())
    ['datetime', 'hour', 'day_of_week', ..., 'is_weekend']
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


def create_sequences(data, n_steps_in=5, n_steps_out=5):
    """
    Create sequences for LSTM training: n_steps_in -> n_steps_out.
    
    Parameters:
    -----------
    data : np.ndarray
        Array of features with shape (n_samples, n_features)
    n_steps_in : int
        Number of time steps to look back (default: 5)
    n_steps_out : int
        Number of time steps to predict (default: 5)
    
    Returns:
    --------
    X : np.ndarray
        Input sequences with shape (n_sequences, n_steps_in, n_features)
    y : np.ndarray
        Output sequences with shape (n_sequences, n_steps_out)
        Contains only the first feature (PM2.5 values)
    
    Notes:
    ------
    - Each sequence is created by sliding a window over the data
    - Output contains only PM2.5 values (first column) for prediction
    - Number of sequences = len(data) - n_steps_in - n_steps_out + 1
    
    Examples:
    ---------
    >>> data = np.random.rand(100, 3)  # 100 samples, 3 features
    >>> X, y = create_sequences(data, n_steps_in=5, n_steps_out=5)
    >>> print(X.shape, y.shape)
    (91, 5, 3) (91, 5)
    """
    X, y = [], []
    
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        # Input sequence: all features
        X.append(data[i:i + n_steps_in])
        # Output sequence: only PM2.5 values (first column)
        y.append(data[i + n_steps_in:i + n_steps_in + n_steps_out, 0])
    
    return np.array(X), np.array(y)


def denormalize_pm25(values, scaler, n_features):
    """
    Denormalize PM2.5 values (first feature) back to original scale.
    
    Parameters:
    -----------
    values : np.ndarray
        1D array of normalized PM2.5 values
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted scaler used for normalization
    n_features : int
        Total number of features used in scaling
    
    Returns:
    --------
    np.ndarray
        Denormalized PM2.5 values in original scale
    
    Notes:
    ------
    This function handles the case where only PM2.5 values (first feature)
    need to be denormalized, but the scaler was fitted on all features.
    
    Examples:
    ---------
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler()
    >>> features = np.random.rand(100, 3)
    >>> features_scaled = scaler.fit_transform(features)
    >>> pm25_scaled = features_scaled[:, 0]
    >>> pm25_original = denormalize_pm25(pm25_scaled, scaler, 3)
    """
    # Create a dummy array with all features
    dummy = np.zeros((values.shape[0], n_features))
    dummy[:, 0] = values  # PM2.5 is the first feature
    # Inverse transform
    denorm = scaler.inverse_transform(dummy)
    return denorm[:, 0]
