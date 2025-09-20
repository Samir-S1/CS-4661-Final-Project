# Interactive Financial LSTM Prediction with User Configuration
# This script allows you to configure all LSTM parameters interactively

import os
import sys
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def validate_execution():
    """Validate that the script is being run correctly"""
    # Check if the script was called with arguments that look like file paths
    if len(sys.argv) > 1:
        print("⚠️  WARNING: It looks like this script was called with command line arguments.")
        print("   This script is designed to be run interactively without arguments.")
        print("   Please run it as: python interactive_financial_lstm.py")
        print("   Or in VS Code: Run Python File in Terminal")
        return False
    return True

def generate_model_id(config):
    """Generate a unique model ID based on configuration"""
    # Create a string representation of ALL key configuration elements
    id_components = [
        config['ticker'],
        f"h{config['hidden_size']}",
        f"l{config['num_layers']}",
        f"d{config['dropout_rate']:.1f}",
        f"seq{config['sequence_length']}",
        f"lr{config['learning_rate']:.4f}".replace('0.', ''),
        f"bs{config['batch_size']}",
        f"ep{config['num_epochs']}",
        f"opt{config['optimizer'][:3]}",  # adam, sgd, rms
        f"sched{1 if config.get('use_scheduler', False) else 0}",
        f"split{config['train_split']:.1f}".replace('0.', ''),
        f"feat{len(config['features'])}",
        "_".join(sorted([f.replace('_', '') for f in config['features']]))[:20]  # Truncate long feature names
    ]
    
    # Add scheduler parameters if used
    if config.get('use_scheduler', False):
        id_components.extend([
            f"sf{config.get('scheduler_factor', 0.5):.1f}".replace('0.', ''),
            f"sp{config.get('scheduler_patience', 10)}"
        ])
    
    # Create a hash of the configuration for uniqueness
    config_str = "_".join(id_components)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
    
    # Return human-readable ID with hash
    model_id = f"lstm_{config['ticker']}_h{config['hidden_size']}_l{config['num_layers']}_f{len(config['features'])}_{config_hash}"
    return model_id

def get_user_config():
    """Get all configuration parameters from user input"""
    print("=" * 60)
    print("INTERACTIVE FINANCIAL LSTM CONFIGURATION")
    print("=" * 60)
    
    config = {}
    
    # Check for existing models
    print("\n🔍 MODEL SELECTION:")
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if model_files:
        print("Found existing models:")
        for i, model_file in enumerate(model_files, 1):
            # Try to load model config to show details
            try:
                model_path = os.path.join(models_dir, model_file)
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                saved_config = checkpoint['config']
                features_str = f"{len(saved_config['features'])} features"
                
                # Get model ID if available
                model_id = checkpoint.get('model_id', model_file.replace('.pth', ''))
                
                print(f"  {i}. {model_id}")
                print(f"     📊 Ticker: {saved_config['ticker']} | Hidden: {saved_config['hidden_size']} | Layers: {saved_config['num_layers']} | {features_str}")
                print(f"     🎯 Features: {', '.join(saved_config['features'][:3])}{'...' if len(saved_config['features']) > 3 else ''}")
            except Exception as e:
                print(f"  {i}. {model_file} (unable to read details)")
        print(f"  {len(model_files) + 1}. Train new model")
        
        choice = input(f"\nSelect option (1-{len(model_files) + 1}, default: {len(model_files) + 1}): ").strip()
        
        if choice and choice.isdigit() and 1 <= int(choice) <= len(model_files):
            selected_model = model_files[int(choice) - 1]
            print(f"Loading model: {selected_model}")
            config['load_model'] = os.path.join(models_dir, selected_model)
            
            # For loaded models, only ask for evaluation period
            print("\n📊 EVALUATION PERIOD:")
            config['start_date'] = input("Enter start date for evaluation (YYYY-MM-DD, default: 2020-01-01): ").strip() or "2020-01-01"
            config['end_date'] = input("Enter end date for evaluation (YYYY-MM-DD, default: 2024-12-01): ").strip() or "2024-12-01"
            print(f"Will evaluate model on data from {config['start_date']} to {config['end_date']}")
            
            return config
        else:
            print("Training new model...")
            config['load_model'] = None
    else:
        print("No existing models found. Will train new model.")
        config['load_model'] = None
    
    # Stock selection
    print("\n📈 STOCK SELECTION:")
    ticker_input = input("Enter stock ticker symbol (default: AAPL): ").strip().upper()
    
    # Validate ticker input - remove any invalid characters
    if ticker_input:
        # Remove any file path characters or command characters that might have leaked through
        invalid_chars = ['/', '\\', ':', '"', '&', '.PY', '.EXE']
        for char in invalid_chars:
            ticker_input = ticker_input.replace(char, '')
        
        # Only keep valid ticker characters (letters and dots)
        ticker_input = ''.join(c for c in ticker_input if c.isalpha() or c == '.')
        
        # If nothing valid remains, use default
        if not ticker_input:
            ticker_input = "AAPL"
    else:
        ticker_input = "AAPL"
    
    config['ticker'] = ticker_input
    print(f"Using ticker: {config['ticker']}")
    config['start_date'] = input("Enter start date (YYYY-MM-DD, default: 2020-01-01): ").strip() or "2020-01-01"
    config['end_date'] = input("Enter end date (YYYY-MM-DD, default: 2024-12-01): ").strip() or "2024-12-01"
    
    # Data preprocessing
    print("\n📊 DATA PREPROCESSING:")
    config['sequence_length'] = int(input("Sequence length - days to look back (default: 60): ").strip() or "60")
    config['train_split'] = float(input("Training data split ratio (default: 0.8): ").strip() or "0.8")
    
    # Model architecture
    print("\n🧠 LSTM MODEL ARCHITECTURE:")
    config['hidden_size'] = int(input("Hidden layer size (default: 64): ").strip() or "64")
    config['num_layers'] = int(input("Number of LSTM layers (default: 2): ").strip() or "2")
    config['dropout_rate'] = float(input("Dropout rate (0.0-0.8, default: 0.2): ").strip() or "0.2")
    
    # Training parameters
    print("\n🏋️ TRAINING PARAMETERS:")
    config['batch_size'] = int(input("Batch size (default: 32): ").strip() or "32")
    config['learning_rate'] = float(input("Learning rate (default: 0.001): ").strip() or "0.001")
    config['num_epochs'] = int(input("Number of epochs (default: 100): ").strip() or "100")
    
    # Optimizer selection
    print("\n⚙️ OPTIMIZER SELECTION:")
    print("1. Adam (default)")
    print("2. SGD")
    print("3. RMSprop")
    optimizer_choice = input("Choose optimizer (1-3, default: 1): ").strip() or "1"
    config['optimizer'] = {'1': 'adam', '2': 'sgd', '3': 'rmsprop'}[optimizer_choice]
    
    # Learning rate scheduler
    print("\n📉 LEARNING RATE SCHEDULER:")
    use_scheduler = input("Use learning rate scheduler? (y/N): ").strip().lower()
    config['use_scheduler'] = use_scheduler in ['y', 'yes']
    
    if config['use_scheduler']:
        config['scheduler_factor'] = float(input("Scheduler reduction factor (default: 0.5): ").strip() or "0.5")
        config['scheduler_patience'] = int(input("Scheduler patience epochs (default: 10): ").strip() or "10")
    
    # Features selection
    print("\n🎯 FEATURE SELECTION:")
    print("Available features:")
    all_features = ['Close_Price', 'Volume', 'High_Low_Pct', 'Open_Close_Pct', 
                   'MA_5', 'MA_10', 'MA_20', 'Volatility']
    for i, feature in enumerate(all_features, 1):
        print(f"{i}. {feature}")
    
    feature_choice = input("Select features (comma-separated numbers, default: all): ").strip()
    if feature_choice:
        selected_indices = [int(x.strip()) - 1 for x in feature_choice.split(',')]
        config['features'] = [all_features[i] for i in selected_indices]
    else:
        config['features'] = all_features
    
    # Display configuration
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY:")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key}: {value}")
    
    confirm = input("\nProceed with this configuration? (Y/n): ").strip().lower()
    if confirm in ['n', 'no']:
        print("Configuration cancelled.")
        return None
    
    return config

def load_saved_model(model_path, eval_config=None):
    """Load a previously saved model and its configuration"""
    try:
        print(f"📁 Loading model from {model_path}...")
        
        # Load the saved data with weights_only=False for sklearn objects
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        saved_config = checkpoint['config']
        scaler = checkpoint['scaler']
        model_id = checkpoint.get('model_id', 'legacy_model')
        
        print(f"✅ Loaded model: {model_id}")
        print(f"   - Ticker: {saved_config['ticker']}")
        print(f"   - Hidden size: {saved_config['hidden_size']}")
        print(f"   - Layers: {saved_config['num_layers']}")
        print(f"   - Features ({len(saved_config['features'])}): {', '.join(saved_config['features'])}")
        print(f"   - Dropout: {saved_config['dropout_rate']}")
        print(f"   - Sequence length: {saved_config['sequence_length']}")
        
        # Use evaluation config dates if provided, otherwise use saved config dates
        download_config = saved_config.copy()
        if eval_config and 'start_date' in eval_config:
            download_config.update(eval_config)
            print(f"   - Using custom evaluation period: {eval_config['start_date']} to {eval_config['end_date']}")
        
        # Download fresh data for the same ticker
        print(f"\n📥 Downloading fresh {saved_config['ticker']} data...")
        df_features, df_raw = download_and_preprocess_data(download_config)
        
        # Create the model with saved configuration
        input_size = len(saved_config['features'])
        model = FinancialLSTM(
            input_size=input_size,
            hidden_size=saved_config['hidden_size'],
            num_layers=saved_config['num_layers'],
            dropout=saved_config['dropout_rate']
        )
        
        # Load the trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        
        return model, saved_config, scaler, df_features, df_raw
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None, None, None, None

class StockDataset(Dataset):
    def __init__(self, data, sequence_length, target_column=0):
        self.data = data
        self.sequence_length = sequence_length
        self.target_column = target_column
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length, self.target_column]
        return torch.FloatTensor(sequence), torch.FloatTensor([target])

class FinancialLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        super(FinancialLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        predictions = self.linear(last_output)
        
        return predictions

def download_and_preprocess_data(config):
    """Download and preprocess stock data"""
    print(f"\n📥 Downloading {config['ticker']} data...")
    
    stock_data = yf.download(config['ticker'], start=config['start_date'], end=config['end_date'])
    
    if stock_data.empty:
        raise ValueError(f"No data found for ticker {config['ticker']}")
    
    # Create features
    df = stock_data.copy()
    df['Close_Price'] = df['Close']
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Price_Change'].rolling(window=10).std()
    
    df = df.dropna()
    df_features = df[config['features']].copy()
    
    print(f"✅ Data shape: {df_features.shape}")
    print(f"✅ Features: {config['features']}")
    
    return df_features, df

def create_datasets(df_features, config):
    """Create train/test datasets"""
    print(f"\n🔄 Creating datasets...")
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_features.values)
    
    # Split data
    train_size = int(len(scaled_data) * config['train_split'])
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    # Create datasets
    train_dataset = StockDataset(train_data, config['sequence_length'])
    test_dataset = StockDataset(test_data, config['sequence_length'])
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset, scaler

def create_model_and_optimizer(config):
    """Create model and optimizer based on configuration"""
    print(f"\n🏗️ Building model...")
    
    input_size = len(config['features'])
    model = FinancialLSTM(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout_rate']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'])
    
    # Create scheduler if requested
    scheduler = None
    if config['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', 
            factor=config['scheduler_factor'], 
            patience=config['scheduler_patience'], 
            verbose=True
        )
    
    criterion = nn.MSELoss()
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✅ Using {config['optimizer']} optimizer")
    print(f"✅ Device: {device}")
    
    return model, optimizer, scheduler, criterion, device

def train_model_function(model, train_loader, criterion, optimizer, device):
    """Training function"""
    model.train()
    total_loss = 0
    
    for sequences, targets in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_model_function(model, test_loader, criterion, device):
    """Validation function"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, device, config):
    """Main training loop"""
    print(f"\n🏋️ Starting training for {config['num_epochs']} epochs...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config['num_epochs']):
        train_loss = train_model_function(model, train_loader, criterion, optimizer, device)
        val_loss = validate_model_function(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if scheduler:
            scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    print("✅ Training completed!")
    return train_losses, val_losses

def evaluate_and_visualize(model, test_loader, scaler, config, train_losses, val_losses, device):
    """Evaluate model and create visualizations"""
    print(f"\n📊 Evaluating model...")
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    # Inverse transform
    dummy_pred = np.zeros((len(predictions), len(config['features'])))
    dummy_actual = np.zeros((len(actuals), len(config['features'])))
    dummy_pred[:, 0] = predictions.flatten()
    dummy_actual[:, 0] = actuals.flatten()
    
    pred_prices = scaler.inverse_transform(dummy_pred)[:, 0]
    actual_prices = scaler.inverse_transform(dummy_actual)[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(actual_prices, pred_prices)
    mae = mean_absolute_error(actual_prices, pred_prices)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
    
    print("📈 MODEL PERFORMANCE METRICS:")
    print("=" * 40)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Accuracy: {100 - mape:.2f}%")
    
    # Create comprehensive multi-timeframe visualizations
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle(f'{config["ticker"]} LSTM Prediction Results - Multi-Timeframe Analysis', fontsize=16)
    
    # Training loss (only if we have training data)
    if train_losses and val_losses:
        axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    else:
        # For loaded models, show model info instead
        axes[0, 0].text(0.5, 0.5, f'Loaded Pre-trained Model\n\nArchitecture:\n- Hidden Size: {config["hidden_size"]}\n- Layers: {config["num_layers"]}\n- Dropout: {config["dropout_rate"]}\n- Features: {len(config["features"])}\n- Total Data Points: {len(actual_prices)}', 
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[0, 0].set_title('Model Information')
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axis('off')
    
    # Multi-timeframe price analysis
    total_days = len(actual_prices)
    
    # Time window 1: Last 50 days (short-term)
    last_50 = min(50, total_days)
    if last_50 > 0:
        axes[0, 1].plot(range(total_days-last_50, total_days), actual_prices[-last_50:], 
                       label='Actual', color='green', linewidth=2)
        axes[0, 1].plot(range(total_days-last_50, total_days), pred_prices[-last_50:], 
                       label='Predicted', color='orange', linewidth=2)
        axes[0, 1].set_title(f'Short-term: Last {last_50} Days')
        axes[0, 1].set_xlabel('Days from Start')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Time window 2: Last 250 days (medium-term, ~1 trading year)
    last_250 = min(250, total_days)
    if last_250 > 0:
        step_250 = max(1, last_250 // 100)  # Subsample if too many points
        indices_250 = range(total_days-last_250, total_days, step_250)
        axes[1, 0].plot(indices_250, actual_prices[-last_250::step_250], 
                       label='Actual', color='green', linewidth=2)
        axes[1, 0].plot(indices_250, pred_prices[-last_250::step_250], 
                       label='Predicted', color='orange', linewidth=2)
        axes[1, 0].set_title(f'Medium-term: Last {last_250} Days (~1 Year)')
        axes[1, 0].set_xlabel('Days from Start')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Time window 3: Full dataset (long-term)
    step_all = max(1, total_days // 200)  # Subsample for clarity
    indices_all = range(0, total_days, step_all)
    axes[1, 1].plot(indices_all, actual_prices[::step_all], 
                   label='Actual', color='green', linewidth=2)
    axes[1, 1].plot(indices_all, pred_prices[::step_all], 
                   label='Predicted', color='orange', linewidth=2)
    axes[1, 1].set_title(f'Long-term: Full Dataset ({total_days} Days)')
    axes[1, 1].set_xlabel('Days from Start')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Performance metrics by time period
    def calc_metrics(actual, predicted, period_name):
        if len(actual) == 0:
            return {}
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        accuracy = 100 - mape  # Accuracy as complement of MAPE
        return {
            'Period': period_name,
            'RMSE': f'{rmse:.2f}',
            'MAE': f'{mae:.2f}',
            'MAPE': f'{mape:.2f}%',
            'Accuracy': f'{accuracy:.2f}%'
        }
    
    # Calculate metrics for different periods
    metrics_data = []
    if last_50 > 0:
        metrics_data.append(calc_metrics(actual_prices[-last_50:], pred_prices[-last_50:], f'Short ({last_50}d)'))
    if last_250 > 0:
        metrics_data.append(calc_metrics(actual_prices[-last_250:], pred_prices[-last_250:], f'Medium ({last_250}d)'))
    metrics_data.append(calc_metrics(actual_prices, pred_prices, f'Full ({total_days}d)'))
    
    # Display metrics table
    if metrics_data:
        metrics_text = "Performance by Time Period:\n\n"
        metrics_text += f"{'Period':<15} {'RMSE':<8} {'MAE':<8} {'MAPE':<8} {'Accuracy':<9}\n"
        metrics_text += "-" * 58 + "\n"
        for metric in metrics_data:
            # Ensure period name fits in 15 characters
            period_name = metric['Period']
            if len(period_name) > 15:
                period_name = period_name[:12] + "..."
            metrics_text += f"{period_name:<15} {metric['RMSE']:<8} {metric['MAE']:<8} {metric['MAPE']:<8} {metric['Accuracy']:<9}\n"
        
        axes[2, 0].text(0.05, 0.95, metrics_text, transform=axes[2, 0].transAxes, 
                       fontfamily='monospace', fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[2, 0].set_title('Performance Metrics by Time Period')
        axes[2, 0].set_xlim(0, 1)
        axes[2, 0].set_ylim(0, 1)
        axes[2, 0].axis('off')
    
    # Error distribution and scatter plot
    errors = actual_prices - pred_prices
    axes[2, 1].hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[2, 1].set_xlabel('Prediction Error ($)')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].set_title(f'Error Distribution (Mean: ${np.mean(errors):.2f}, Std: ${np.std(errors):.2f})')
    axes[2, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[2, 1].axvline(x=np.mean(errors), color='orange', linestyle='-', linewidth=2, label=f'Mean: ${np.mean(errors):.2f}')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return pred_prices, actual_prices

def main():
    """Main execution function"""
    try:
        # Validate execution method
        if not validate_execution():
            input("\nPress Enter to continue anyway, or Ctrl+C to exit...")
        
        # Get user configuration
        config = get_user_config()
        if config is None:
            return
        
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Check if we should load an existing model or train a new one
        if config.get('load_model'):
            # Extract evaluation config (start_date, end_date) for loaded models
            eval_config = {k: v for k, v in config.items() if k in ['start_date', 'end_date']}
            
            # Load existing model
            model, saved_config, scaler, df_features, df_raw = load_saved_model(config['load_model'], eval_config)
            if model is None:
                print("Failed to load model. Exiting.")
                return
            
            # Use the saved config but keep the evaluation dates
            config = saved_config.copy()
            config.update(eval_config)
            config['load_model'] = True  # Mark as loaded model
            
            # Create test dataset with loaded data
            train_dataset, test_dataset, _ = create_datasets(df_features, config)
            
            # Initialize empty training losses for loaded models
            train_losses = []
            val_losses = []
            
        else:
            # Train new model - original flow
            # Download and preprocess data
            df_features, df_raw = download_and_preprocess_data(config)
            
            # Create datasets
            train_dataset, test_dataset, scaler = create_datasets(df_features, config)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
            
            # Create model and optimizer
            model, optimizer, scheduler, criterion, device = create_model_and_optimizer(config)
            
            # Train model
            print("\n🏋️ Training new model...")
            train_losses, val_losses = train_model(
                model, train_loader, test_loader, optimizer, scheduler, criterion, device, config
            )
            
        # Create test data loader (for both loaded and trained models)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Set device for evaluation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Evaluate and visualize
        if config.get('load_model'):
            print(f"\n📊 Evaluating loaded model on fresh {config['ticker']} data...")
        
        pred_prices, actual_prices = evaluate_and_visualize(
            model, test_loader, scaler, config, train_losses, val_losses, device
        )
        
        print(f"\n🎉 Analysis complete for {config['ticker']}!")
        
        # Ask if user wants to save the model (only for newly trained models)
        if not config.get('load_model'):
            save_model = input("\nSave trained model? (y/N): ").strip().lower()
            if save_model in ['y', 'yes']:
                # Get the directory where this script is located
                script_dir = os.path.dirname(os.path.abspath(__file__))
                models_dir = os.path.join(script_dir, "models")
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                
                # Generate unique model ID
                model_id = generate_model_id(config)
                model_path = os.path.join(models_dir, f"{model_id}.pth")
                
                # Check if model already exists and ask for confirmation
                if os.path.exists(model_path):
                    overwrite = input(f"⚠️  Model {model_id} already exists. Overwrite? (y/N): ").strip().lower()
                    if overwrite not in ['y', 'yes']:
                        print("Model not saved.")
                        return
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'scaler': scaler,
                    'model_id': model_id
                }, model_path)
                print(f"✅ Model saved as {model_path}")
                print(f"🆔 Model ID: {model_id}")
                print(f"📊 Features: {', '.join(config['features'])}")
        else:
            print("📁 Used existing model - no saving needed.")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()