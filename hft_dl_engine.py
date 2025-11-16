# hft_dl_engine_clean.py
# Requires: python 3.9+, torch, numpy, sklearn, onnx, onnxruntime
# pip install torch numpy scikit-learn onnx onnxruntime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
import onnxruntime as ort
import os
import warnings

# ------------------ Config ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 64
FEATURES = 12
BATCH = 512
EPOCHS = 10
LR = 1e-3
HIDDEN = 64
# --------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning)  # suppress ONNX dynamic axes warnings

class TickDataset(Dataset):
    def __init__(self, X, y_reg, y_clf):
        self.X = X.astype(np.float32)
        self.y_reg = y_reg.astype(np.float32)
        self.y_clf = y_clf.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_clf[idx]

class TinyConvNet(nn.Module):
    def __init__(self, seq_len, features, hidden=HIDDEN):
        super().__init__()
        self.conv1 = nn.Conv1d(features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_shared = nn.Sequential(
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
        self.reg_head = nn.Linear(hidden//2, 1)
        self.clf_head = nn.Linear(hidden//2, 3)

    def forward(self, x):
        # (B, seq_len, features) -> (B, features, seq_len)
        x = x.permute(0,2,1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        shared = self.fc_shared(x)
        reg = self.reg_head(shared).squeeze(-1)
        clf_logits = self.clf_head(shared)
        return reg, clf_logits

def train_loop(model, opt, loader, device):
    model.train()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    total = 0
    for Xb, y_reg, y_clf in loader:
        Xb, y_reg, y_clf = Xb.to(device), y_reg.to(device), y_clf.to(device)
        opt.zero_grad()
        reg_pred, clf_logits = model(Xb)
        loss = mse(reg_pred, y_reg) + 0.5*ce(clf_logits, y_clf)
        loss.backward()
        opt.step()
        total += loss.item() * Xb.size(0)
    return total / len(loader.dataset)

def eval_loop(model, loader, device):
    model.eval()
    mse = nn.MSELoss(reduction='sum')
    ce = nn.CrossEntropyLoss(reduction='sum')
    total_mse, total_ce, correct, total = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for Xb, y_reg, y_clf in loader:
            Xb, y_reg, y_clf = Xb.to(device), y_reg.to(device), y_clf.to(device)
            reg_pred, clf_logits = model(Xb)
            total_mse += mse(reg_pred, y_reg).item()
            total_ce += ce(clf_logits, y_clf).item()
            pred = torch.argmax(clf_logits, dim=1)
            correct += (pred == y_clf).sum().item()
            total += Xb.size(0)
    return {"mse": total_mse / total, "ce": total_ce / total, "acc": correct / total}

def main_train():
    # Load data
    X = np.load("X.npy")
    y_reg = np.load("y_reg.npy")
    y_clf = np.load("y_clf.npy")

    split_idx = int(len(X) * 0.8)
    X_tr, X_val = X[:split_idx], X[split_idx:]
    ytr_reg, yval_reg = y_reg[:split_idx], y_reg[split_idx:]
    ytr_clf, yval_clf = y_clf[:split_idx], y_clf[split_idx:]

    tr_loader = DataLoader(TickDataset(X_tr, ytr_reg, ytr_clf), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TickDataset(X_val, yval_reg, yval_clf), batch_size=BATCH, shuffle=False)

    model = TinyConvNet(SEQ_LEN, FEATURES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    best_val = float('inf')
    for ep in range(EPOCHS):
        train_loss = train_loop(model, opt, tr_loader, DEVICE)
        val_metrics = eval_loop(model, val_loader, DEVICE)
        print(f"Epoch {ep}: train_loss {train_loss:.6f}, val_mse {val_metrics['mse']:.6e}, acc {val_metrics['acc']:.4f}")
        if val_metrics['mse'] < best_val:
            best_val = val_metrics['mse']
            # Save weights only (avoids FutureWarning)
            torch.save(model.state_dict(), "best_model.pth", _use_new_zipfile_serialization=True)

    # Export ONNX safely on CPU
    dummy = torch.randn(1, SEQ_LEN, FEATURES, device='cpu')
    model.cpu()
    model.load_state_dict(torch.load("best_model.pth", map_location='cpu', weights_only=True))
    model.eval()
    onnx_path = "hft_model.onnx"
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["X"], output_names=["reg", "clf_logits"],
        opset_version=13, dynamic_axes={"X": {0: "batch_size"}}
    )
    print("Exported ONNX ->", onnx_path)

def onnx_infer_example(onnx_path="hft_model.onnx"):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.load("X.npy")[:1].astype(np.float32)
    reg_pred, clf_logits = sess.run(None, {"X": x})
    print("Reg pred:", reg_pred, "clf logits:", clf_logits)

if __name__ == "__main__":
    if not os.path.exists("X.npy"):
        print("No sample 'X.npy' found. Please prepare X.npy, y_reg.npy, y_clf.npy and re-run.")
    else:
        main_train()
        onnx_infer_example()