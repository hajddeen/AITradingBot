# -*- coding: utf-8 -*-
# real_time_hybrid.py
# Hybrid inference: ONNX CUDA / ONNX CPU / PyTorch GPU
# Requires: numpy, onnxruntime (optional), torch (optional)

import asyncio
import concurrent.futures
from collections import deque
from datetime import datetime
import time
import numpy as np
import os

# ============================================================
# >>> LOGGING
# ============================================================
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# >>> FORCE FLAGS (IMPORTANT)
# ============================================================
FORCE_PYTORCH_GPU = True      # <-- forces PyTorch GPU and disables ONNX
FORCE_CPU_ONLY = False        # <-- forces CPU everywhere
LOG_INPUT_OUTPUT = True
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_BATCH = []
LOG_BATCH_SIZE = 200

def log_to_csv(row: dict):
    date = datetime.utcnow().strftime("%Y-%m-%d")
    csv_path = f"{LOG_DIR}/ticks_{date}.csv"
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

def log_to_parquet_batch(rows: list):
    if not rows:
        return
    date = datetime.utcnow().strftime("%Y-%m-%d")
    pq_path = f"{LOG_DIR}/ticks_{date}.parquet"
    table = pa.Table.from_pylist(rows)
    if os.path.exists(pq_path):
        existing = pq.read_table(pq_path)
        table = pa.concat_tables([existing, table])
    pq.write_table(table, pq_path)

def log_event(row: dict):
    LOG_BATCH.append(row)
    if len(LOG_BATCH) >= LOG_BATCH_SIZE:
        log_to_parquet_batch(LOG_BATCH)
        LOG_BATCH.clear()
    log_to_csv(row)
# ============================================================


# -------------------- CONFIG --------------------
ONNX_PATH = "hft_model.onnx"
PT_WEIGHTS = "best_model.pth"

SEQ_LEN = 64
FEATURES = 12
REG_THRESHOLD = 0.0005

FEATURE_MEAN = np.zeros(FEATURES, dtype=np.float32)
FEATURE_STD = np.ones(FEATURES, dtype=np.float32)

SIMULATED_RATE_HZ = 1000
QUEUE_MAXSIZE = 20000
INFER_WORKERS = 3
# ------------------------------------------------

def timestamp_ms():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ========================================================================
# BACKEND SELECTION
# ========================================================================
sess = None
onnx_used = False
pt_used = False
pt_model = None
pt_device = None

# --------------------
# OPTION A — FORCE CPU
# --------------------
if FORCE_CPU_ONLY:
    print(f"{timestamp_ms()} | FORCE_CPU_ONLY=True. Disabling GPU and ONNX CUDA completely")
    FORCE_PYTORCH_GPU = False

# --------------------
# OPTION B — FORCE PYTORCH GPU ONLY
# --------------------
if FORCE_PYTORCH_GPU:
    print(f"{timestamp_ms()} | FORCE_PYTORCH_GPU=True. Skipping all ONNX initialization")
    sess = None
    onnx_used = False
else:
    # ----------------------------------------------------------
    # Try ONNX (only when not forcing PyTorch)
    # ----------------------------------------------------------
    try:
        import onnxruntime as ort

        # Try full CUDA first
        try:
            print(f"{timestamp_ms()} | Attempting ONNX GPU session...")
            sess = ort.InferenceSession(
                ONNX_PATH,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            print(f"{timestamp_ms()} | ONNX GPU session created. Providers: {sess.get_providers()}")
            onnx_used = True

        except Exception as e_gpu:
            print(f"{timestamp_ms()} | ONNX CUDA session failed: {e_gpu}")
            print(f"{timestamp_ms()} | Falling back to ONNX CPU...")
            try:
                sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
                print(f"{timestamp_ms()} | ONNX CPU session created.")
                onnx_used = True
            except Exception as e_cpu:
                print(f"{timestamp_ms()} | ONNX CPU session ALSO failed: {e_cpu}")
                sess = None

    except Exception as e_import:
        print(f"{timestamp_ms()} | onnxruntime import failed: {e_import}")
        sess = None
        onnx_used = False


# ----------------------------------------------------------
# If ONNX unavailable load PyTorch instead
# ----------------------------------------------------------
if sess is None:
    try:
        import torch

        try:
            from hft_dl_engine import TinyConvNet, SEQ_LEN as _SEQ, FEATURES as _FEAT, DEVICE as _DEV
            print(f"{timestamp_ms()} | Imported model from hft_dl_engine.py")
        except Exception:
            from AITradingBot import TinyConvNet, SEQ_LEN as _SEQ, FEATURES as _FEAT, DEVICE as _DEV
            print(f"{timestamp_ms()} | Imported model from AITradingBot.py")

        pt_device = "cuda" if (torch.cuda.is_available() and not FORCE_CPU_ONLY) else "cpu"

        pt_model = TinyConvNet(SEQ_LEN, FEATURES).to(pt_device)
        pt_model.load_state_dict(torch.load(PT_WEIGHTS, map_location=pt_device, weights_only=True))
        pt_model.eval()

        pt_used = True
        print(f"{timestamp_ms()} | PyTorch model loaded on {pt_device}")

    except Exception as e_pt:
        print(f"{timestamp_ms()} | PyTorch backend failed: {e_pt}")
        raise RuntimeError("No usable inference backend available.")

# ========================================================================
# END BACKEND SELECTION
# ========================================================================



# -------------------- Feature extraction --------------------
def normalize(x: np.ndarray):
    return (x - FEATURE_MEAN) / (FEATURE_STD + 1e-9)

def extract_features(tick: dict) -> np.ndarray:
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
    out = session.run(None, {"X": X_input})
    reg = float(np.asarray(out[0]).reshape(-1)[0])
    clf_logits = np.asarray(out[1])
    clf = int(np.argmax(clf_logits, axis=1)[0])
    return reg, clf, clf_logits

def pytorch_infer(model, X_input_np: np.ndarray):
    import torch
    x = torch.from_numpy(X_input_np).to(next(model.parameters()).device)
    with torch.no_grad():
        reg_t, clf_logits = model(x)
        reg = float(reg_t.cpu().numpy().reshape(-1)[0])
        logits = clf_logits.cpu().numpy()
        clf = int(np.argmax(logits, axis=1)[0])
    return reg, clf, logits


# -------------------- Async pipeline --------------------
tick_buffer = deque(maxlen=SEQ_LEN)

async def tick_producer(queue: asyncio.Queue, rate_hz=SIMULATED_RATE_HZ):
    interval = 1.0 / rate_hz
    while True:
        tick = {
            "bid1_price": 100.0 + np.random.randn()*0.0005,
            "ask1_price": 100.0 + 0.001 + np.random.randn()*0.0005,
            "bid1_size": np.random.randint(1, 200),
            "ask1_size": np.random.randint(1, 200),
            "trade_flow": np.random.randn(),
            "vwap": 100.0 + np.random.randn()*0.0005,
            "bid2_price": 99.999 + np.random.randn()*0.0005,
            "ask2_price": 100.001 + np.random.randn()*0.0005,
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
            f_norm = normalize(f)
            tick_buffer.append(f_norm)

            if len(tick_buffer) == SEQ_LEN:

                X_input = np.array(tick_buffer, dtype=np.float32)[None, :, :]
                t0 = time.perf_counter()

                if sess is not None:
                    reg_pred, clf_pred, logits = await loop.run_in_executor(
                        executor, onnx_infer, sess, X_input
                    )
                else:
                    reg_pred, clf_pred, logits = await loop.run_in_executor(
                        executor, pytorch_infer, pt_model, X_input
                    )

                latency_ms = (time.perf_counter() - t0) * 1000

                if reg_pred > REG_THRESHOLD and clf_pred == 2:
                    signal = "BUY"
                elif reg_pred < -REG_THRESHOLD and clf_pred == 0:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                print(f"{timestamp_ms()} | reg={reg_pred:.6f} clf={['DOWN','NEUTRAL','UP'][clf_pred]:7s} -> {signal}")

                # LOGGING
                log_row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "raw_tick": tick,
                    "features": f.tolist(),
                    "features_norm": f_norm.tolist(),
                    "sequence_hash": hash(tuple(X_input.reshape(-1).tolist())),
                    "regression": float(reg_pred),
                    "classification": int(clf_pred),
                    "logits": logits.reshape(-1).tolist(),
                    "signal": signal,
                    "latency_ms": float(latency_ms)
                }
                log_event(log_row)

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
    print(f"{timestamp_ms()} | Starting hybrid engine. ONNX used: {onnx_used}, PyTorch used: {pt_used}")
    asyncio.run(main())
