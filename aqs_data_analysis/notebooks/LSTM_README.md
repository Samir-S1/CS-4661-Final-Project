# LSTM-based PM2.5 Prediction Model

## Overview

This notebook (`lstm_airnow2.ipynb`) implements an LSTM (Long Short-Term Memory) neural network model to predict PM2.5 air quality levels using AirNow/AQS data.

## Key Features

### 1. **Data Processing with Trust Values**
- **Forward Filling**: Missing data points are filled using the most recent observation
- **Backward Filling**: Gaps before the first observation are filled using future observations
- **Trust Value Calculation**: Each data point is assigned a trust value (0-1) based on:
  - Original observations: trust = 1.0
  - Filled values: trust decays exponentially based on distance from original observation
  - Decay rate: 0.9 per hour (configurable)

### 2. **Temporal Feature Engineering**
The model includes rich temporal features:
- **Hour of day** (cyclical encoding: sin/cos)
- **Day of week** (cyclical encoding: sin/cos)
- **Month** (cyclical encoding: sin/cos)
- **Weekend indicator** (binary)
- **Day of year**

Cyclical encoding ensures that the model understands that hour 23 is close to hour 0, December is close to January, etc.

### 3. **LSTM Model Architecture**
- **Input**: 5 consecutive hourly measurements (with 9 features each)
- **Output**: 5 future hourly PM2.5 predictions
- **Architecture**:
  - LSTM layer (64 units) with return sequences
  - Dropout (0.2)
  - LSTM layer (32 units)
  - Dropout (0.2)
  - Dense layer (32 units, ReLU activation)
  - Dropout (0.2)
  - Output layer (5 units, linear activation)

### 4. **Visualizations**
- **Trust-based color coding**: Data points colored by trust value (red=low trust, green=high trust)
- **Fill method visualization**: Different colors for original, forward-filled, and backward-filled data
- **Training history**: Loss and MAE curves for training and validation
- **Prediction comparison**: Multiple examples of actual vs. predicted values
- **Scatter plots**: Prediction quality across different time horizons (1, 3, and 5 hours ahead)
- **Time series plots**: Continuous predictions over test period

## Prerequisites

### Data Requirements
You need an AQS database file located at:
```
../data/aqs_data.db
```

The database should contain a table named `hourly_88101_2020` with PM2.5 measurements.

To create this database, follow the instructions in the main `aqs_data_analysis/README.md` and run:
```bash
python scripts/load_data.py
```

### Python Dependencies
Install required packages:
```bash
pip install -r ../requirements.txt
```

Or install manually:
```bash
pip install pandas numpy matplotlib plotly scikit-learn tensorflow jupyter ipykernel seaborn
```

## Usage

### Running the Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook lstm_airnow2.ipynb
   ```

2. **Run all cells** sequentially (Kernel → Restart & Run All)

3. **Expected runtime**: 5-15 minutes depending on hardware and data size

### Customization Options

#### Change Monitoring Site
In cell 3, modify the SQL query:
```python
# Change '0010' to your desired site number
WHERE "State Name" = 'California' AND "Site Num" = '0019'
```

#### Adjust Trust Decay Rate
In cell 9, modify the decay rate:
```python
# Default is 0.9 (90% trust retention per hour)
df_complete = calculate_trust_values(df_complete, 'Sample Measurement', decay_rate=0.8)
```

#### Modify Prediction Horizon
In cell 18, change the sequence parameters:
```python
# Change n_steps_in and n_steps_out
n_steps_in = 10  # Use 10 hours of history
n_steps_out = 3  # Predict 3 hours ahead
```

#### Adjust Model Architecture
In cell 21, modify the `build_lstm_model` function:
```python
# Example: Add more LSTM layers or change units
LSTM(128, activation='tanh', return_sequences=True, ...)
```

#### Training Parameters
In cell 23, adjust training settings:
```python
# Increase epochs for longer training
epochs=100,
# Change batch size
batch_size=64,
# Adjust early stopping patience
patience=20
```

## Output and Results

### Model Performance Metrics
The notebook calculates and displays:
- **MSE (Mean Squared Error)**: Overall loss metric
- **MAE (Mean Absolute Error)**: Average prediction error in µg/m³
- **RMSE (Root Mean Squared Error)**: Standard deviation of errors
- **R² Score**: Proportion of variance explained by the model

Metrics are provided for:
- Each prediction horizon (1-5 hours ahead)
- Overall performance across all horizons

### Visualizations Generated

1. **Trust-based scatter plot**: PM2.5 over time with color-coded trust values
2. **Fill method comparison**: Visualization showing original vs. filled data
3. **Trust values over time**: Timeline of data quality
4. **Training history**: Loss and MAE curves
5. **Prediction examples**: 5 example sequences with actual vs. predicted
6. **Time series predictions**: Continuous predictions for 1-hour and 5-hour horizons
7. **Scatter plots**: Prediction accuracy for 1, 3, and 5-hour horizons

## Understanding the Results

### Trust Values
- **Trust = 1.0**: Original measurements (most reliable)
- **Trust = 0.9**: 1 hour from nearest observation
- **Trust = 0.81**: 2 hours from nearest observation (0.9²)
- **Trust = 0.73**: 3 hours from nearest observation (0.9³)
- And so on...

### Good Model Performance
- **MAE < 5 µg/m³**: Excellent for short-term predictions (1-2 hours)
- **MAE < 10 µg/m³**: Good for medium-term predictions (3-4 hours)
- **R² > 0.7**: Strong predictive power
- Predictions should follow actual values in the time series plots
- Scatter plots should cluster along the diagonal line

### Interpreting Predictions
- **1-hour ahead**: Usually most accurate (MAE lowest)
- **5-hours ahead**: Less accurate but still useful for planning
- Look for systematic biases (e.g., over/under-prediction at high/low values)
- Check if the model captures diurnal patterns (daily cycles)

## Troubleshooting

### Database Not Found
**Error**: `Database not found at ../data/aqs_data.db`
**Solution**: Run `python scripts/load_data.py` to create the database from CSV files

### Out of Memory
**Error**: Memory error during training
**Solutions**:
- Reduce batch size: `batch_size=16`
- Use less data: Modify SQL query to select shorter time period
- Use sampling: `LIMIT 50000` in SQL query

### Model Not Training
**Issue**: Loss not decreasing
**Solutions**:
- Check data quality: Verify PM2.5 values are reasonable
- Normalize data: Ensure scaler is working correctly
- Reduce model complexity: Fewer LSTM units
- Increase learning rate or change optimizer

### Poor Predictions
**Issue**: High MAE or low R²
**Solutions**:
- Increase training data: Use more historical data
- Add more features: Weather data, day of year, etc.
- Adjust model architecture: More layers or units
- Tune hyperparameters: Learning rate, dropout, etc.
- Check for data leakage: Ensure train/val/test split is correct

## Model Limitations

1. **Data Quality Dependent**: Performance degrades with poor data quality or large gaps
2. **Site-Specific**: Model trained on one site may not generalize to others
3. **No External Factors**: Doesn't account for weather, traffic, wildfires, etc.
4. **Hourly Granularity**: Cannot predict sub-hourly variations
5. **Stationarity Assumption**: May not handle sudden changes well (e.g., wildfire events)

## Future Improvements

Potential enhancements:
1. **Multi-site modeling**: Train on data from multiple locations
2. **External features**: Incorporate weather data, traffic patterns
3. **Attention mechanisms**: Use Transformer architecture for better long-range dependencies
4. **Ensemble methods**: Combine multiple models for better predictions
5. **Online learning**: Update model with new data continuously
6. **Uncertainty quantification**: Provide confidence intervals for predictions

## References

- EPA Air Quality System: https://www.epa.gov/aqs
- LSTM Networks: Hochreiter & Schmidhuber (1997)
- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- PM2.5 Health Standards: https://www.epa.gov/pm-pollution

## Support

For issues or questions:
1. Check the main project README
2. Review the data loading scripts in `../scripts/`
3. Verify your environment matches the requirements
4. Check that your data follows the expected schema

## License

This notebook is part of the CS-4661 Final Project.
