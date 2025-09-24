# src/04_train_pytorch.py
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load processed data
train_np = np.load(PROCESSED_DIR / "train.npz")
test_np = np.load(PROCESSED_DIR / "test.npz")
X_train = train_np["X"].astype(np.float32)
y_train = train_np["y"].astype(np.float32)
X_test = test_np["X"].astype(np.float32)
y_test = test_np["y"].astype(np.float32)

INPUT_DIM = X_train.shape[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, "Input dim:", INPUT_DIM)

# Model
class StudentNet(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x.squeeze(-1)

# Datasets
batch_size = 32
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Instantiate
model = StudentNet(INPUT_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Training loop with simple early stopping
best_val_loss = float("inf")
epochs = 100
patience = 10
patience_counter = 0

for epoch in range(1, epochs+1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # validation
    model.eval()
    val_losses = []
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())
            preds_all.extend(preds.detach().cpu().numpy().tolist())
            targets_all.extend(yb.detach().cpu().numpy().tolist())

    avg_train = float(np.mean(train_losses))
    avg_val = float(np.mean(val_losses))
    # Manual RMSE (instead of squared=False)
    mse_val = mean_squared_error(targets_all, preds_all)
    rmse_val = np.sqrt(mse_val)
    print(f"Epoch {epoch} train_loss={avg_train:.4f} val_loss={avg_val:.4f} val_RMSE={rmse_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_counter = 0
        torch.save(model.state_dict(), MODELS_DIR / "student_state_dict.pt")
        print(" Saved best model state_dict.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# After training: load best model and evaluate on test set fully
best_model = StudentNet(INPUT_DIM).to(DEVICE)
best_model.load_state_dict(torch.load(MODELS_DIR / "student_state_dict.pt", map_location=DEVICE))
best_model.eval()

# Predict on test
with torch.no_grad():
    X_test_t = torch.from_numpy(X_test).to(DEVICE)
    preds_test = best_model(X_test_t).cpu().numpy()

# Final test metrics
mse = mean_squared_error(y_test, preds_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, preds_test)
r2 = r2_score(y_test, preds_test)

print("\nFinal Test Metrics (PyTorch ANN):")
print(f" RMSE: {rmse:.4f}")
print(f" MAE:  {mae:.4f}")
print(f" R2:   {r2:.4f}")

# Save predictions with original features
raw_df = pd.read_csv(ROOT / "data" / "raw" / "student-mat.csv", sep=';')
X_df = raw_df.drop(columns=['G3'])
y_df = raw_df['G3'].values.astype(float)
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

out_df = X_test_df.reset_index(drop=True).copy()
out_df["G3"] = y_test_df
out_df["ann_pred"] = preds_test
out_df["ann_error"] = out_df["ann_pred"] - out_df["G3"]
out_df.to_csv(PROCESSED_DIR / "ann_predictions.csv", index=False)
print("Saved ANN predictions to:", PROCESSED_DIR / "ann_predictions.csv")
print("Saved best model to:", MODELS_DIR / "student_state_dict.pt")
