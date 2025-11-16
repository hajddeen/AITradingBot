# real_time_hft.py
import numpy as np
import torch
from collections import deque
from hft_dl_engine import TinyConvNet, SEQ_LEN, FEATURES, DEVICE

# --- Configuration ---
REG_THRESHOLD = 0.0005  # microprice return threshold
clf_labels = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

# --- Load trained model ---
model = TinyConvNet(SEQ_LEN, FEATURES).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE, weights_only=True))
model.eval()

# --- Tick buffer ---
tick_buffer = deque(maxlen=SEQ_LEN)

# --- Feature extraction function ---
def extract_features(tick):
    """
    tick: dict with raw tick data
    returns: numpy array of shape (FEATURES,)
    """
    bid1 = tick['bid1_price']
    ask1 = tick['ask1_price']
    bid1_size = tick['bid1_size']
    ask1_size = tick['ask1_size']
    
    mid = (bid1 + ask1) / 2
    spread = ask1 - bid1
    imbalance = bid1_size / (bid1_size + ask1_size + 1e-6)
    
    # Example feature vector (expand as needed)
    features = np.array([
        bid1, ask1, bid1_size, ask1_size,
        spread, mid, imbalance,
        tick.get('trade_flow', 0),
        tick.get('vwap', mid),
        tick.get('bid2_price', 0),
        tick.get('ask2_price', 0),
        tick.get('extra_feature', 0)
    ], dtype=np.float32)
    return features

# --- Simulate incoming tick stream ---
def get_next_tick():
    """
    Replace this with real API call to exchange/broker
    """
    return {
        "bid1_price": np.random.random(),
        "ask1_price": np.random.random() + 0.001,
        "bid1_size": np.random.randint(1, 100),
        "ask1_size": np.random.randint(1, 100),
        "trade_flow": np.random.random(),
        "vwap": np.random.random(),
        "bid2_price": np.random.random(),
        "ask2_price": np.random.random(),
        "extra_feature": np.random.random()
    }

# --- Generate trading signal ---
def generate_signal(reg_pred, clf_pred):
    """
    Simple strategy example
    """
    if reg_pred > REG_THRESHOLD and clf_pred == 2:  # UP
        return "BUY"
    elif reg_pred < -REG_THRESHOLD and clf_pred == 0:  # DOWN
        return "SELL"
    else:
        return "HOLD"

# --- Main real-time loop ---
while True:
    tick = get_next_tick()
    features = extract_features(tick)
    tick_buffer.append(features)

    if len(tick_buffer) == SEQ_LEN:
        X_input = np.array(tick_buffer)[None, :, :]  # shape (1, SEQ_LEN, FEATURES)
        X_tensor = torch.from_numpy(X_input).to(DEVICE)

        with torch.no_grad():
            reg_pred, clf_logits = model(X_tensor)
            reg_pred = reg_pred.item()
            clf_pred = torch.argmax(clf_logits, dim=1).item()
        
        signal = generate_signal(reg_pred, clf_pred)
        print(f"Reg: {reg_pred:.6f}, Clf: {clf_labels[clf_pred]}, Signal: {signal}")