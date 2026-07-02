import os
import argparse
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import mlflow
import mlflow.pytorch

# Set experiment name (equivalent to off_test.py pattern)
mlflow.set_experiment("Seizure Recognition")


def load_epileptic_seizure_data(csv_path, seq_len=256, test_size=0.2):
    """Load real seizure data from CSV or generate synthetic."""
    if os.path.exists(csv_path):
        print(f"Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        # Assuming last column is label, rest are features
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)
        # Normalize each sample to mean 0, std 1
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        # Reshape to (batch, channels=1, seq_len) if needed
        if X.shape[1] != seq_len:
            print(f"Warning: CSV has {X.shape[1]} features, expected {seq_len}")
        X = X.reshape(X.shape[0], 1, -1)
    else:
        print(f"CSV not found at {csv_path}. Generating synthetic data.")
        X, y = generate_synthetic_eeg(n_samples=1000, seq_len=seq_len)
    
    # Split into train/val
    split = int(len(X) * (1 - test_size))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    return X_train, y_train, X_val, y_val


def generate_synthetic_eeg(n_samples=1000, seq_len=256, seizure_prob=0.2, seed=42):
    """Generate synthetic EEG signals with seizure patterns."""
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, size=(n_samples, 1, seq_len)).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        if rng.rand() < seizure_prob:
            max_width = max(1, min(20, seq_len - 20))
            width = rng.randint(4, max_width + 1)
            pos_low = 10
            pos_high = seq_len - 10 - width + 1
            pos = pos_low if pos_high <= pos_low else rng.randint(pos_low, pos_high)
            amp = rng.uniform(3, 8)
            window = np.hanning(width)
            X[i, 0, pos:pos+width] += amp * window
            y[i] = 1
    return X, y


class SimpleEEGNet(nn.Module):
    def __init__(self, seq_len=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(args):
    """Train seizure recognition model with autologging support."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    csv_path = args.csv_path if args.csv_path else "/home/aiuser/app/Epileptic Seizure Recognition.csv"
    X_train, y_train, X_val, y_val = load_epileptic_seizure_data(csv_path, seq_len=args.seq_len)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    model = SimpleEEGNet(seq_len=args.seq_len).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    batch_size = args.batch_size
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train phase
        model.train()
        perm = np.random.permutation(len(X_train))
        train_losses = []
        
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx].astype(np.float32)).to(device)
            
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                xb = torch.from_numpy(X_val[i:i+batch_size]).to(device)
                yb = torch.from_numpy(y_val[i:i+batch_size].astype(np.float32)).to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb).item()
                val_losses.append(loss)
        
        avg_val_loss = np.mean(val_losses)
        
        # Log results
        print(f"Epoch {epoch}/{args.epochs} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")
        
        # Always log metrics (MLflow active context auto-logs if run is active)
        mlflow.log_metric("train_loss", float(avg_train_loss), step=epoch)
        mlflow.log_metric("val_loss", float(avg_val_loss), step=epoch)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            best_model_path = os.path.join(args.output_dir, "eeg_model_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved to {best_model_path}")
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    final_model_path = os.path.join(args.output_dir, "eeg_model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    result = {
        "model_path_final": final_model_path,
        "model_path_best": os.path.join(args.output_dir, "eeg_model_best.pt"),
        "best_val_loss": float(best_val_loss),
    }
    
    # Autolog will handle model logging; no explicit call needed
    if args.mlflow:
        print("✓ Model auto-logged to MLflow (via pytorch.autolog)")
        result["mlflow_artifact_path"] = "eeg_model"
    
    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"mlflow_enabled": args.mlflow, **result}, f, indent=2)
    print(f"Metadata saved to {meta_path}")
    
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Seizure recognition with PyTorch + MLflow")
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--seq-len", dest="seq_len", type=int, default=256, help="Sequence length")
    p.add_argument("--output-dir", dest="output_dir", default="models", help="Output directory for weights")
    p.add_argument("--csv-path", dest="csv_path", default=None, help="Path to CSV data (optional)")
    p.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    p.add_argument("--cpu", action="store_true", help="Force CPU usage")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.mlflow:
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "(not set)")
        print(f"MLflow tracking URI: {mlflow_uri}")
        # Enable autologging for PyTorch
        mlflow.pytorch.autolog()
        mlflow.start_run()
        mlflow.log_params({
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
        })
    
    result = train(args)
    
    if args.mlflow:
        mlflow.end_run()
        print("MLflow run ended.")
    
    print("\n✓ Training complete. Results:")
    for k, v in result.items():
        print(f"  {k}: {v}")
