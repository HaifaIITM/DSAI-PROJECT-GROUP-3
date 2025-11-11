# Hybrid Model Save/Load Guide

The HybridESNRidge model now supports saving and loading trained models to disk for later inference.

## Quick Start

### 1. Train and Save Model

```python
# Option A: Using evaluate_hybrid_model.py (recommended)
python evaluate_hybrid_model.py

# Option B: Using pipeline directly
from src.pipeline import run_baseline

result = run_baseline(
    model_name="hybrid",
    fold_id=0,
    horizon="target_h20",
    save_model=True  # Enable saving
)
```

### 2. Load and Use Saved Model

```python
from src.models.hybrid_esn_ridge import HybridESNRidge
import numpy as np

# Load saved model
model = HybridESNRidge.load("data/experiments/hybrid/fold_0/model_target_h20")

# Generate predictions
X_new = np.array([...])  # Your feature matrix
predictions = model.predict(X_new)
```

### 3. Demo Script

Run the complete demo to see save/load in action:

```bash
# First, train and save a model
python evaluate_hybrid_model.py

# Then, load and use it
python load_hybrid_model_demo.py
```

## What Gets Saved?

When you save a HybridESNRidge model, the following files are created:

```
data/experiments/hybrid/fold_<id>/model_<horizon>/
├── config.json          # Model hyperparameters
├── esn_weights.npz      # ESN reservoir weights and state
└── ridge_model.pkl      # Ridge regression model
```

### config.json
Contains all hyperparameters needed to reconstruct the model:
- `hidden_size`: ESN reservoir size
- `spectral_radius`: ESN spectral radius
- `leak_rate`: ESN leak rate
- `esn_alpha`: ESN regularization
- `ridge_alpha`: Ridge regularization
- `washout`: ESN warmup steps
- `seed`: Random seed

### esn_weights.npz
Contains the trained ESN weights:
- `W_in`: Input weights matrix
- `W`: Reservoir recurrent weights
- `W_out`: Readout weights
- `last_state_`: Last reservoir state
- `input_dim_`, `output_dim_`: Dimensions
- `_fitted`: Training status flag

### ridge_model.pkl
The complete scikit-learn Ridge model (uses pickle serialization).

## API Reference

### Save Model

```python
model.save(save_dir: str)
```

**Parameters:**
- `save_dir`: Directory path where model files will be saved

**Example:**
```python
model = HybridESNRidge(hidden_size=1600)
model.fit(X_train, y_train)
model.save("my_models/hybrid_v1")
```

### Load Model

```python
model = HybridESNRidge.load(save_dir: str)
```

**Parameters:**
- `save_dir`: Directory path where model files were saved

**Returns:**
- Fully initialized HybridESNRidge model ready for prediction

**Example:**
```python
model = HybridESNRidge.load("my_models/hybrid_v1")
predictions = model.predict(X_test)
```

## Integration with Training Pipeline

### Using run_baseline()

```python
from src.pipeline import run_baseline

result = run_baseline(
    model_name="hybrid",
    fold_id=0,
    horizon="target_h20",
    save_model=True  # Add this parameter
)
# Model saved to: data/experiments/hybrid/fold_0/model_target_h20/
```

### Using run_experiment()

```python
from src.train.runner import run_experiment

result = run_experiment(
    model_name="hybrid",
    fold_id=0,
    horizon="target_h20",
    model_kwargs={"hidden_size": 1600, "esn_alpha": 0.3, "ridge_alpha": 1.0},
    save_model=True  # Add this parameter
)
# Model saved to: data/experiments/hybrid/<exp_id>/fold_0/model_target_h20/
```

## Use Cases

### 1. Production Deployment
Save your best model after training, then load it in a production environment without retraining:

```python
# Training phase (one-time)
model = train_hybrid_model()
model.save("production_models/hybrid_h20")

# Production phase (load once at startup)
model = HybridESNRidge.load("production_models/hybrid_h20")

# Inference (fast, repeated)
for new_data in data_stream:
    prediction = model.predict(new_data)
```

### 2. Model Versioning
Save different model versions for comparison:

```python
# Train multiple configurations
for config in configs:
    model = HybridESNRidge(**config)
    model.fit(X_train, y_train)
    model.save(f"models/hybrid_{config['hidden_size']}_{config['esn_alpha']}")

# Load and compare
model_v1 = HybridESNRidge.load("models/hybrid_1600_0.3")
model_v2 = HybridESNRidge.load("models/hybrid_2000_0.5")
```

### 3. Cross-Validation
Save models from each fold for ensemble predictions:

```python
# Train and save all folds
for fold_id in range(9):
    result = run_baseline("hybrid", fold_id=fold_id, horizon="target_h20", save_model=True)

# Load and ensemble
models = [
    HybridESNRidge.load(f"data/experiments/hybrid/fold_{i}/model_target_h20")
    for i in range(9)
]
predictions = np.mean([m.predict(X_test) for m in models], axis=0)
```

## File Size

Typical saved model size:
- **config.json**: < 1 KB
- **esn_weights.npz**: ~50-100 MB (for hidden_size=1600)
- **ridge_model.pkl**: < 1 MB
- **Total**: ~50-100 MB per model

For storage optimization, compress with:
```bash
tar -czf hybrid_model.tar.gz data/experiments/hybrid/fold_0/model_target_h20/
```

## Notes

- The model saves its complete internal state, so predictions from loaded models will exactly match the original trained model
- Make sure to use the same feature preprocessing pipeline when generating predictions
- The saved model is portable and can be loaded on different machines (same Python/numpy versions recommended)
- For best results, use the same 38 features (10 technical + 28 headline embeddings) as during training

## Troubleshooting

### Model Not Found
```python
FileNotFoundError: Model not found at: data/experiments/hybrid/fold_0/model_target_h20
```
**Solution**: Train and save the model first using `evaluate_hybrid_model.py`

### Dimension Mismatch
```python
ValueError: X has 20 features, expected 38
```
**Solution**: Ensure your input data has the same number of features as training data

### Prediction Differs from Training
**Solution**: Make sure you're using `continue_state=True` (default) and loading from the same training run

## Support

For questions or issues:
1. Check the demo script: `load_hybrid_model_demo.py`
2. Review the model code: `src/models/hybrid_esn_ridge.py`
3. See training examples: `evaluate_hybrid_model.py`

