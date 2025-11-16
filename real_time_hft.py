# real_time_hft.py
import numpy as np
import torch
from collections import deque
import time

# Import model definitions and constants from your training script.
# If your training file is named AITradingBot.py use that name here:
from AITradingBot import TinyConvNet, SEQ_LEN, FEATURES, DEVICE
# OR if you renamed it to hft_dl_engine.py use:
# from hft_dl_engine import TinyConvNet, SEQ_LEN, FEATURES, DEVICE

# --- config / simple thresholds
REG_THRESHOLD = 0.0005
clf_labels = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

# --- load trained model (PyTorch)
model = TinyConvNet(SEQ_LEN, FEATURES).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE, weights_only=True))
model.eval()

# --- sliding buffer for last SEQ_LEN feature vectors
tick_buffer = deque(maxlen=SEQ_LEN)

# --- user must keep same normalization used in training
# placeholder mean/std (replace with your real training scalers)
FEATURE_MEAN = np.zeros(FEATURES, dtype=np.float32)
FEATURE_STD = np.ones(FEATURES, dtype=np.float32)

def normalize(x):
    return (x - FEATURE_MEAN) / (FEATURE_STD + 1e-9)

def extract_features(tick):
    bid1 = tick['bid1_price']
    ask1 = tick['ask1_price']
    bid1_size = tick['bid1_size']
    ask1_size = tick['ask1_size']
    mid = (bid1 + ask1) / 2
    spread = ask1 - bid1
    imbalance = bid1_size / (bid1_size + ask1_size + 1e-9)
    features = np.array([
        bid1, ask1, bid1_size, ask1_size,
        spread, mid, imbalance,
        tick.get('trade_flow', 0.0),
        tick.get('vwap', mid),
        tick.get('bid2_price', 0.0),
        tick.get('ask2_price', 0.0),
        tick.get('extra_feature', 0.0)
    ], dtype=np.float32)
    return features

def get_next_tick():
    # Replace with real exchange/broker streaming API
    return {
        "bid1_price": 100.0 + np.random.randn()*0.0005,
        "ask1_price": 100.0 + 0.001 + np.random.randn()*0.0005,
        "bid1_size": np.random.randint(1,200),
        "ask1_size": np.random.randint(1,200),
        "trade_flow": np.random.randn(),
        "vwap": 100.0 + np.random.randn()*0.0005,
        "bid2_price": 99.999 + np.random.randn()*0.0005,
        "ask2_price": 100.001 + np.random.randn()*0.0005,
        "extra_feature": np.random.randn()
    }

def generate_signal(reg_pred, clf_pred):
    if reg_pred > REG_THRESHOLD and clf_pred == 2:
        return "BUY"
    elif reg_pred < -REG_THRESHOLD and clf_pred == 0:
        return "SELL"
    else:
        return "HOLD"

print("Starting sync real-time loop (Ctrl-C to stop)...")
try:
    while True:
        tick = get_next_tick()
        f = extract_features(tick)
        f = normalize(f)
        tick_buffer.append(f)

        if len(tick_buffer) == SEQ_LEN:
            X_input = np.array(tick_buffer, dtype=np.float32)[None, :, :]  # shape (1, SEQ_LEN, FEATURES)
            xt = torch.from_numpy(X_input).to(DEVICE)
            with torch.no_grad():
                reg_pred, clf_logits = model(xt)
                reg_pred = reg_pred.item()
                clf_pred = int(torch.argmax(clf_logits, dim=1).item())

            signal = generate_signal(reg_pred, clf_pred)
            ts = time.strftime("%H:%M:%S")
            print(f"{ts} | reg={reg_pred:.6f} clf={clf_labels[clf_pred]:7s} -> {signal}")
        # adapt sleep to desired tick ingestion rate (simulate high-frequency by small sleep)
        time.sleep(0.001)  # 1 ms loop for simulation
except KeyboardInterrupt:
    print("Stopped by user.")