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
# >>> FORCE FLAGS
# ============================================================
FORCE_PYTORCH_GPU = True      # <-- forces PyTorch GPU and disables ONNX
FORCE_CPU_ONLY = False        # <-- forces CPU everywhere
SIMULATED_RATE_HZ = 1000
QUEUE_MAXSIZE = 20000
INFER_WORKERS = 3
LOG_PARQUET_BATCH = 200
LOG_QUEUE_MAX = 20000
LOG_DROP_ON_FULL = True
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# >>> NON-BLOCKING BACKGROUND LOGGER
# ============================================================
import csv
import threading
import queue
import pyarrow as pa
import pyarrow.parquet as pq

LOG_Q = queue.Queue(maxsize=LOG_QUEUE_MAX)
_LOG_THREAD = None
_LOG_THREAD_STOP = threading.Event()
_parquet_buffer = []

def _csv_path_for_date(dt: datetime):
    date = dt.strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"ticks_{date}.csv")

def _parquet_path_for_date(dt: datetime):
    date = dt.strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"ticks_{date}.parquet")

def enqueue_log(row: dict):
    """Called from hot path. Non-blocking (may drop rows)."""
    try:
        LOG_Q.put_nowait(row)
    except queue.Full:
        if LOG_DROP_ON_FULL:
            return
        else:
            LOG_Q.put(row)

def _write_csv_row(path: str, row: dict):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def _flush_parquet_buffer(target_path: str, buffer_rows: list):
    if not buffer_rows:
        return
    table = pa.Table.from_pylist(buffer_rows)
    if not os.path.exists(target_path):
        pq.write_table(table, target_path)
    else:
        with pq.ParquetWriter(target_path, table.schema, use_dictionary=True, compression='snappy') as writer:
            writer.write_table(table)

def _logger_thread_func():
    global _parquet_buffer
    while not _LOG_THREAD_STOP.is_set() or not LOG_Q.empty():
        try:
            row = LOG_Q.get(timeout=0.5)
        except queue.Empty:
            if _LOG_THREAD_STOP.is_set() and _parquet_buffer:
                try:
                    ppath = _parquet_path_for_date(datetime.utcnow())
                    _flush_parquet_buffer(ppath, _parquet_buffer)
                finally:
                    _parquet_buffer.clear()
            continue

        try:
            csv_path = _csv_path_for_date(datetime.utcnow())
            _write_csv_row(csv_path, row)
        except Exception:
            pass

        try:
            _parquet_buffer.append(row)
        except Exception:
            pass

        if len(_parquet_buffer) >= LOG_PARQUET_BATCH:
            try:
                ppath = _parquet_path_for_date(datetime.utcnow())
                _flush_parquet_buffer(ppath, _parquet_buffer)
            except Exception:
                pass
            _parquet_buffer.clear()
        LOG_Q.task_done()

    # final flush
    if _parquet_buffer:
        try:
            ppath = _parquet_path_for_date(datetime.utcnow())
            _flush_parquet_buffer(ppath, _parquet_buffer)
        except Exception:
            pass
        _parquet_buffer.clear()

def start_log_worker():
    global _LOG_THREAD, _LOG_THREAD_STOP
    if _LOG_THREAD is None or not _LOG_THREAD.is_alive():
        _LOG_THREAD_STOP.clear()
        _LOG_THREAD = threading.Thread(target=_logger_thread_func, name="hft-logger", daemon=True)
        _LOG_THREAD.start()

def stop_log_worker(timeout: float = 5.0):
    global _LOG_THREAD, _LOG_THREAD_STOP
    _LOG_THREAD_STOP.set()
    if _LOG_THREAD is not None:
        _LOG_THREAD.join(timeout=timeout)
    while not LOG_Q.empty():
        try:
            row = LOG_Q.get_nowait()
            try:
                csv_path = _csv_path_for_date(datetime.utcnow())
                _write_csv_row(csv_path, row)
            except Exception:
                pass
            LOG_Q.task_done()
        except queue.Empty:
            break

def flush_sync():
    LOG_Q.join()
    stop_log_worker(timeout=10.0)

# ============================================================
# -------------------- CONFIG --------------------
ONNX_PATH = "hft_model.onnx"
PT_WEIGHTS = "best_model.pth"
SEQ_LEN = 64
FEATURES = 12
REG_THRESHOLD = 0.0005
FEATURE_MEAN = np.zeros(FEATURES, dtype=np.float32)
FEATURE_STD = np.ones(FEATURES, dtype=np.float32)
tick_buffer = deque(maxlen=SEQ_LEN)

def timestamp_ms():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

# ============================================================
# -------------------- BACKEND SELECTION --------------------
sess = None
onnx_used = False
pt_used = False
pt_model = None
pt_device = None

if FORCE_CPU_ONLY:
    print(f"{timestamp_ms()} | FORCE_CPU_ONLY=True. Disabling GPU and ONNX CUDA completely")
    FORCE_PYTORCH_GPU = False

if FORCE_PYTORCH_GPU:
    print(f"{timestamp_ms()} | FORCE_PYTORCH_GPU=True. Skipping all ONNX initialization")
    sess = None
    onnx_used = False
else:
    try:
        import onnxruntime as ort
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

# ============================================================
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

# ============================================================
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

# ============================================================
# -------------------- Async pipeline --------------------
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

                # Logging
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
                enqueue_log(log_row)

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
        print("CTRL+C detected. Flushing logs")

# ============================================================
# -------------------- MAIN --------------------
if __name__ == "__main__":
    start_log_worker()
    print(f"{timestamp_ms()} | Starting hybrid engine. ONNX used: {onnx_used}, PyTorch used: {pt_used}")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("CTRL+C detected. Shutting down")
    finally:
        print("Flushing logs and stopping logger")
        flush_sync()
        print("Shutdown complete.")
