# -*- coding: utf-8 -*-
# real_time_hybrid.py
# Hybrid inference: ONNX CUDA ONNX CPU PyTorch GPU
# Requires: numpy, onnxruntime (optional), torch (optional)
# Run: python real_time_hybrid.py

import asyncio
import concurrent.futures
from collections import deque
from datetime import datetime
import time
import numpy as np
import os

# -------------------- CONFIG --------------------
ONNX_PATH = "hft_model.onnx"
PT_WEIGHTS = "best_model.pth"

SEQ_LEN = 64
FEATURES = 12
REG_THRESHOLD = 0.0005

# Normalization placeholders (replace with training scalers)
FEATURE_MEAN = np.zeros(FEATURES, dtype=np.float32)
FEATURE_STD = np.ones(FEATURES, dtype=np.float32)

# Tick producer rate (Hz) for simulation — replace with real feed callback integration
SIMULATED_RATE_HZ = 1000

# Async queue max size (backpressure)
QUEUE_MAXSIZE = 20000

# Threadpool workers for inference
INFER_WORKERS = 3
# ------------------------------------------------

def timestamp_ms():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

# -------------------- Backend selection --------------------
sess = None
onnx_used = False
pt_used = False
pt_model = None
pt_device = None

# Try ONNXRuntime
try:
    import onnxruntime as ort
    # Try GPU provider first (if available)
    try:
        sess = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        print(f"{timestamp_ms()} | ONNX session providers: {sess.get_providers()}")
        onnx_used = True
    except Exception as e_gpu:
        print(f"{timestamp_ms()} | ONNX CUDA session failed: {e_gpu}")
        # Try CPU provider
        try:
            sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
            print(f"{timestamp_ms()} | ONNX CPU session created.")
            onnx_used = True
        except Exception as e_cpu:
            print(f"{timestamp_ms()} | ONNX CPU session failed: {e_cpu}")
            sess = None
except Exception as e_import:
    print(f"{timestamp_ms()} | onnxruntime import failed or onnx not available: {e_import}")
    sess = None

# If ONNX unavailable, fallback to PyTorch
if sess is None:
    try:
        import torch
        # Try to import your model class from likely module names
        try:
            from hft_dl_engine import TinyConvNet, SEQ_LEN as _SEQ, FEATURES as _FEAT, DEVICE as _DEV
            print(f"{timestamp_ms()} | Imported model from hft_dl_engine.py")
        except Exception:
            try:
                from AITradingBot import TinyConvNet, SEQ_LEN as _SEQ, FEATURES as _FEAT, DEVICE as _DEV
                print(f"{timestamp_ms()} | Imported model from AITradingBot.py")
            except Exception as e_mod:
                raise ImportError("Could not import TinyConvNet from hft_dl_engine.py or AITradingBot.py") from e_mod

        # Ensure seq/features match config (warn if not)
        if _SEQ != SEQ_LEN or _FEAT != FEATURES:
            print(f"{timestamp_ms()} | WARNING: SEQ_LEN/FETCH mismatch: model({_SEQ},{_FEAT}) vs config({SEQ_LEN},{FEATURES})")

        pt_device = "cuda" if torch.cuda.is_available() else "cpu"
        pt_model = TinyConvNet(SEQ_LEN, FEATURES).to(pt_device)
        pt_model.load_state_dict(torch.load(PT_WEIGHTS, map_location=pt_device, weights_only=True))
        pt_model.eval()
        pt_used = True
        print(f"{timestamp_ms()} | PyTorch model loaded on {pt_device}")
    except Exception as e_pt:
        print(f"{timestamp_ms()} | PyTorch fallback failed: {e_pt}")
        raise RuntimeError("No usable inference backend available (ONNX failed and PyTorch fallback failed).")

# -------------------- Feature extraction & normalization --------------------
def normalize(x: np.ndarray):
    return (x - FEATURE_MEAN) / (FEATURE_STD + 1e-9)

def extract_features(tick: dict) -> np.ndarray:
    # Replace or extend with your actual engineered features (must match training)
    bid1 = tick['bid1_price']
    ask1 = tick['ask1_price']
    bid1_size = tick['bid1_size']
    ask1_size = tick['ask1_size']
    mid = (bid1 + ask1) / 2.0
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

# -------------------- Inference wrappers --------------------
def onnx_infer(session, X_input: np.ndarray):
    # X_input shape: (1, SEQ_LEN, FEATURES) dtype=float32
    out = session.run(None, {"X": X_input})
    reg = float(np.asarray(out[0]).reshape(-1)[0])
    clf_logits = np.asarray(out[1])
    clf = int(np.argmax(clf_logits, axis=1)[0])
    return reg, clf

def pytorch_infer(model, X_input_np: np.ndarray):
    import torch
    x = torch.from_numpy(X_input_np).to(next(model.parameters()).device)
    with torch.no_grad():
        reg_t, clf_logits = model(x)
        reg = float(reg_t.cpu().numpy().reshape(-1)[0])
        clf = int(torch.argmax(clf_logits, dim=1).cpu().numpy()[0])
    return reg, clf

# -------------------- Async pipeline --------------------
tick_buffer = deque(maxlen=SEQ_LEN)

async def tick_producer(queue: asyncio.Queue, rate_hz=SIMULATED_RATE_HZ):
    interval = 1.0 / rate_hz
    while True:
        # Replace with your live tick code. This is simulated.
        tick = {
            "bid1_price": 100.0 + np.random.randn() * 0.0005,
            "ask1_price": 100.0 + 0.001 + np.random.randn() * 0.0005,
            "bid1_size": np.random.randint(1, 200),
            "ask1_size": np.random.randint(1, 200),
            "trade_flow": np.random.randn(),
            "vwap": 100.0 + np.random.randn() * 0.0005,
            "bid2_price": 99.999 + np.random.randn() * 0.0005,
            "ask2_price": 100.001 + np.random.randn() * 0.0005,
            "extra_feature": np.random.randn()
        }
        await queue.put(tick)
        await asyncio.sleep(interval)

async def tick_consumer(queue: asyncio.Queue):
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=INFER_WORKERS)
    try:
        while True:
            tick = await queue.get()
            f = extract_features(tick)
            f = normalize(f)
            tick_buffer.append(f)

            if len(tick_buffer) == SEQ_LEN:
                X_input = np.array(tick_buffer, dtype=np.float32)[None, :, :]  # (1,SEQ,FEAT)
                if sess is not None:
                    # ONNX path (run in threadpool)
                    reg_pred, clf_pred = await loop.run_in_executor(executor, onnx_infer, sess, X_input)
                else:
                    # PyTorch path (run in threadpool)
                    reg_pred, clf_pred = await loop.run_in_executor(executor, pytorch_infer, pt_model, X_input)

                # simple decision logic
                if reg_pred > REG_THRESHOLD and clf_pred == 2:
                    signal = "BUY"
                elif reg_pred < -REG_THRESHOLD and clf_pred == 0:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                print(f"{timestamp_ms()} | reg={reg_pred:.6f} clf={['DOWN','NEUTRAL','UP'][clf_pred]:7s} -> {signal}")
            queue.task_done()
    except asyncio.CancelledError:
        executor.shutdown(wait=False)
        raise

async def main():
    q = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    producer = asyncio.create_task(tick_producer(q))
    consumer = asyncio.create_task(tick_consumer(q))
    try:
        await asyncio.gather(producer, consumer)
    except KeyboardInterrupt:
        producer.cancel()
        consumer.cancel()
        await asyncio.gather(producer, consumer, return_exceptions=True)

if __name__ == "__main__":
    # quick checks summary
    print(f"{timestamp_ms()} | Starting hybrid engine. ONNX used: {onnx_used}, PyTorch used: {pt_used}")
    asyncio.run(main())