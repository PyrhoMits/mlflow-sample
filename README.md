# Seizure Recognition with PyTorch + MLflow

Train a CNN-based model for seizure detection from EEG data. The script loads real data from `Epileptic Seizure Recognition.csv` if available, otherwise generates synthetic data.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training locally

```bash
python ml_flow_test.py --epochs 10
```

### 3. Run with MLflow tracking

First start an MLflow server:

```bash
mlflow server --port 5000
```

Then run training:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python ml_flow_test.py --mlflow --epochs 10
```

## Output

- `models/eeg_model_final.pt` — Final trained weights
- `models/eeg_model_best.pt` — Best model (lowest validation loss)
- `models/metadata.json` — Run metadata including losses and MLflow paths

## Arguments

- `--epochs` (default: 10) — Number of training epochs
- `--lr` (default: 0.001) — Learning rate
- `--batch-size` (default: 32) — Batch size
- `--seq-len` (default: 256) — Sequence length
- `--output-dir` (default: `models`) — Output directory
- `--csv-path` — Optional path to CSV file
- `--mlflow` — Enable MLflow logging
- `--cpu` — Force CPU usage

## CSV Format

The script expects a CSV where:
- Rows are samples
- All columns except the last are features (EEG channels/time steps)
- Last column is the binary label (0 or 1)

If the CSV is not found, synthetic data is generated automatically.
# mlflow-sample
# mlflow-sample
