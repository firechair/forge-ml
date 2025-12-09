# How Time Series Forecasting Works

## Overview

This project uses LSTM (Long Short-Term Memory) networks to forecast future values based on historical time series data.

## Architecture

```
Input Sequence (96 timesteps)
    ↓
LSTM Layers (2 layers, 128 hidden units each)
    ↓
Fully Connected Layer
    ↓
Predictions (24 future timesteps)
```

## Training Process

### 1. Data Preparation

```python
# Load data
data = load_dataset("ETTh1")  # Shape: [N, features]

# Split into train/val/test (70/15/15)
train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# Normalize using training mean/std
normalized = (data - mean) / std
```

### 2. Sequence Creation

Sliding window approach:

```
Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]

Sequence 1: [1, 2, 3] → Predict [4, 5]
Sequence 2: [2, 3, 4] → Predict [5, 6]
Sequence 3: [3, 4, 5] → Predict [6, 7]
...
```

Parameters:
- `sequence_length`: 96 (past timesteps used for prediction)
- `prediction_length`: 24 (future timesteps to predict)
- `stride`: 1 (step size between sequences)

### 3. Model Training

```python
for epoch in epochs:
    for sequences, targets in train_loader:
        # Forward pass
        predictions = model(sequences)

        # Calculate loss (MSE)
        loss = MSELoss(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Validate
    val_loss = evaluate(model, val_loader)

    # Save if best
    if val_loss < best_val_loss:
        save_model()
```

Early stopping prevents overfitting.

### 4. Evaluation Metrics

- **MSE (Mean Squared Error)**: Primary loss function
- **MAE (Mean Absolute Error)**: Interpretable error metric

## Prediction Process

```python
# 1. Load trained model
forecaster = load_model("best_model")

# 2. Prepare input sequence
sequence = last_96_timesteps  # Shape: [96, 1]

# 3. Normalize
sequence_norm = (sequence - mean) / std

# 4. Predict
with torch.no_grad():
    predictions_norm = model(sequence_norm)

# 5. Denormalize
predictions = predictions_norm * std + mean

# Result: 24 future timesteps
```

## LSTM Explained

LSTM cells have:
- **Input Gate**: Controls new information flow
- **Forget Gate**: Decides what to discard
- **Output Gate**: Controls output
- **Cell State**: Long-term memory

This allows LSTMs to:
- Learn long-term dependencies
- Avoid vanishing gradients
- Model complex temporal patterns

## Hyperparameters

Key hyperparameters to tune:

1. **hidden_size** (128): LSTM hidden units
   - Larger = more capacity, slower training
   - Smaller = faster, may underfit

2. **num_layers** (2): Stacked LSTM layers
   - More layers = deeper model
   - Too many = overfitting risk

3. **sequence_length** (96): Input window
   - Longer = more context
   - Shorter = faster training

4. **prediction_length** (24): Forecast horizon
   - Longer predictions = harder task
   - Short-term easier than long-term

5. **learning_rate** (0.001): Optimization step size
   - Too high = unstable
   - Too low = slow convergence

## Tips for Better Performance

1. **More data**: Longer time series = better learning
2. **Feature engineering**: Add time features (hour, day, month)
3. **Multivariate**: Use related variables (temperature, humidity, etc.)
4. **Ensembling**: Train multiple models, average predictions
5. **Hyperparameter tuning**: Use grid search or Optuna

## Common Patterns in Time Series

- **Trend**: Long-term increase/decrease
- **Seasonality**: Repeating patterns (daily, weekly, yearly)
- **Cycles**: Non-fixed periodic variations
- **Noise**: Random fluctuations

LSTM can learn all these patterns automatically!
