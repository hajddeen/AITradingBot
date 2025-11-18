# -*- coding: utf-8 -*-
# real_time_hybrid.py
# Hybrid inference with Unified Broker Interface (MT5 backend included)
# Option A logger (daily CSV, hourly Parquet), Prometheus metrics

import asyncio
import concurrent.futures
from collections import deque
from datetime import datetime, timezone
import time
import numpy as np
import os

# ---------------- User flags / config ----------------
BROKER_NAME = "mt5"         # change to "mt5" or others later
BROKER_SYMBOL = "EURUSD"    # symbol for the broker
BROKER_POLL_MS = 5          # polling interval for MT5 backend (ms)
FORCE_PYTORCH_GPU = True
FORCE_CPU_ONLY = False
SIMULATED_RATE_HZ = 1000
QUEUE_MAXSIZE = 20000
INFER_WORKERS = 3

# Logging/persistence config
LOG_DIR = "logs"
JOURNAL_DIR = "journal"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(JOURNAL_DIR, exist_ok=True)
LOG_PARQUET_BATCH = 200
LOG_QUEUE_MAX = 20000
LOG_DROP_ON_FULL = True
PARQUET_COMPRESSION = "snappy"
PROMETHEUS_PORT = 8001

# Model config
ONNX_PATH = "hft_model.onnx"
PT_WEIGHTS = "best_model.pth"
SEQ_LEN = 64
FEATURES = 12
REG_THRESHOLD = 0.0005
FEATURE_MEAN = np.zeros(FEATURES, dtype=np.float32)
FEATURE_STD = np.ones(FEATURES, dtype=np.float32)

# ----------------------------------------------------
tick_buffer = deque(maxlen=SEQ_LEN)

def timestamp_ms():
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]

# ----------------- Prometheus -----------------
from prometheus_client import start_http_server, Gauge, Counter, Summary
PROM_QUEUE_SIZE = Gauge("hft_logger_queue_size", "Number of items queued for logger")
PROM_DROPPED = Counter("hft_logger_dropped_total", "Number of dropped log rows due to full queue")
PROM_WRITTEN_CSV = Counter("hft_logger_written_csv_total", "Number of rows written to CSV")
PROM_WRITTEN_PARQUET = Counter("hft_logger_written_parquet_total", "Number of rows written to Parquet (batched)")
PROM_INFER_LATENCY = Summary("hft_inference_latency_ms", "Inference latency in milliseconds (summary)")

# ----------------- Lightweight background logger -----------------
import csv, threading, queue, json, gzip
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

LOG_Q = queue.Queue(maxsize=LOG_QUEUE_MAX)
_LOG_THREAD = None
_LOG_THREAD_STOP = threading.Event()
_parquet_buffer = []
_current_csv_file = None
_current_csv_date = None
_current_csv_path = None
_journal_write_file = None
_journal_write_path = None
_journal_write_bytes = 0

def _utc_date_str(ts=None):
    if ts is None:
        ts = datetime.utcnow()
    return ts.strftime("%Y-%m-%d")

def _utc_hour_str(ts=None):
    if ts is None:
        ts = datetime.utcnow()
    return ts.strftime("%Y-%m-%d_%H")

def _csv_path_for_date_str(date_str: str):
    return Path(LOG_DIR) / f"ticks_{date_str}.csv"

def _parquet_path_for_hour_str(hour_str: str):
    return Path(LOG_DIR) / f"ticks_{hour_str}.parquet"

def _journal_new_path():
    return Path(JOURNAL_DIR) / f"journal_{int(time.time()*1000)}.jsonl"

def _open_new_journal_if_needed():
    global _journal_write_file, _journal_write_path, _journal_write_bytes
    if _journal_write_file is None or _journal_write_bytes >= (5 * 1024 * 1024):
        existing = sorted(Path(JOURNAL_DIR).glob("journal_*.jsonl"))
        if len(existing) >= 20:
            try:
                existing[0].unlink()
            except Exception:
                pass
        path = _journal_new_path()
        _journal_write_path = path
        _journal_write_file = open(path, "ab")
        _journal_write_bytes = 0

def _journal_append(row: dict):
    global _journal_write_bytes
    try:
        _open_new_journal_if_needed()
        b = (json.dumps(row, default=str) + "\n").encode("utf-8")
        _journal_write_file.write(b)
        _journal_write_file.flush()
        _journal_write_bytes += len(b)
    except Exception:
        pass

def _journal_replay_and_clear():
    files = sorted(Path(JOURNAL_DIR).glob("journal_*.jsonl"))
    for f in files:
        try:
            with open(f, "rb") as fh:
                for line in fh:
                    try:
                        row = json.loads(line.decode("utf-8"))
                        try:
                            LOG_Q.put_nowait(row)
                        except queue.Full:
                            break
                    except Exception:
                        continue
            try:
                f.unlink()
            except Exception:
                pass
        except Exception:
            continue

def _rotate_csv_if_needed():
    global _current_csv_file, _current_csv_date, _current_csv_path
    now_date = _utc_date_str()
    if _current_csv_date != now_date:
        if _current_csv_file:
            try:
                _current_csv_file.flush()
                _current_csv_file.close()
            except Exception:
                pass
        _current_csv_date = now_date
        _current_csv_path = _csv_path_for_date_str(now_date)
        _current_csv_file = open(_current_csv_path, "a", newline="", encoding="utf-8")

def _write_csv_row_handle(row: dict):
    global _current_csv_file, _current_csv_path
    try:
        if _current_csv_file is None:
            _rotate_csv_if_needed()
        writer = csv.DictWriter(_current_csv_file, fieldnames=list(row.keys()))
        if _current_csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        _current_csv_file.flush()
        PROM_WRITTEN_CSV.inc()
        return True
    except Exception:
        return False

def _flush_parquet_buffer_to_path(path: Path, rows: list):
    if not rows:
        return 0
    try:
        table = pa.Table.from_pylist(rows)
        if not path.exists():
            pq.write_table(table, path, compression=PARQUET_COMPRESSION)
        else:
            existing = pq.read_table(path)
            combined = pa.concat_tables([existing, table])
            pq.write_table(combined, path, compression=PARQUET_COMPRESSION)
        PROM_WRITTEN_PARQUET.inc(len(rows))
        return len(rows)
    except Exception:
        return 0

def _logger_thread_func():
    global _parquet_buffer, _current_csv_file, _journal_write_file, _journal_write_bytes
    _journal_replay_and_clear()
    try:
        start_http_server(PROMETHEUS_PORT)
    except Exception:
        pass
    last_rotate_check = time.time()
    while not _LOG_THREAD_STOP.is_set() or not LOG_Q.empty():
        PROM_QUEUE_SIZE.set(LOG_Q.qsize())
        try:
            row = LOG_Q.get(timeout=0.5)
        except queue.Empty:
            now = time.time()
            if now - last_rotate_check > 1.0:
                _rotate_csv_if_needed()
                last_rotate_check = now
            continue

        _rotate_csv_if_needed()
        _journal_append(row)

        try:
            _write_csv_row_handle(row)
        except Exception:
            pass

        try:
            _parquet_buffer.append(row)
        except Exception:
            pass

        if len(_parquet_buffer) >= LOG_PARQUET_BATCH:
            hour_str = _utc_hour_str()
            ppath = _parquet_path_for_hour_str(hour_str)
            try:
                _flush_parquet_buffer_to_path(ppath, _parquet_buffer)
            except Exception:
                pass
            _parquet_buffer.clear()

        LOG_Q.task_done()

    # final flush
    if _parquet_buffer:
        hour_str = _utc_hour_str()
        ppath = _parquet_path_for_hour_str(hour_str)
        try:
            _flush_parquet_buffer_to_path(ppath, _parquet_buffer)
        except Exception:
            pass
        _parquet_buffer.clear()

    try:
        if _current_csv_file:
            _current_csv_file.flush()
            _current_csv_file.close()
    except Exception:
        pass

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
                _rotate_csv_if_needed()
                _write_csv_row_handle(row)
            except Exception:
                pass
            LOG_Q.task_done()
        except queue.Empty:
            break

def flush_sync():
    LOG_Q.join()
    stop_log_worker(timeout=10.0)

def enqueue_log(row: dict):
    try:
        LOG_Q.put_nowait(row)
        return True
    except queue.Full:
        PROM_DROPPED.inc()
        if LOG_DROP_ON_FULL:
            return False
        else:
            LOG_Q.put(row)
            return True

# ---------------- Broker factory and selection ----------------
from broker.factory import get_broker

broker = None
def tick_to_queue_callback(q: asyncio.Queue):
    """
    Returns a callback that can be called from blocking code (broker thread)
    to put ticks into the asyncio queue using loop.call_soon_threadsafe.
    """
    loop = asyncio.get_event_loop()
    def cb(tick: dict):
        # push into asyncio queue in thread-safe manner
        try:
            loop.call_soon_threadsafe(q.put_nowait, tick)
        except Exception:
            # last-resort: try blocking put (should be rare)
            try:
                loop.call_soon_threadsafe(lambda: asyncio.create_task(q.put(tick)))
            except Exception:
                pass
    return cb

# ---------------- Backend selection (unchanged) ----------------
sess = None
onnx_used = False
pt_used = False
pt_model = None
pt_device = None

if FORCE_CPU_ONLY:
    print(f"{timestamp_ms()} | FORCE_CPU_ONLY=True. Disabling GPU and ONNX CUDA completely")
    FORCE_PYTORCH_GPU = False

# If not forcing PyTorch, try ONNX first
if FORCE_PYTORCH_GPU:
    print(f"{timestamp_ms()} | FORCE_PYTORCH_GPU=True. Skipping ONNX initialization (use PyTorch).")
    sess = None
    onnx_used = False
else:
    try:
        import onnxruntime as ort
        try:
            print(f"{timestamp_ms()} | Attempting ONNX GPU session...")
            sess = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            print(f"{timestamp_ms()} | ONNX session providers: {sess.get_providers()}")
            onnx_used = True
        except Exception as e_gpu:
            print(f"{timestamp_ms()} | ONNX GPU failed: {e_gpu}; falling back to CPU")
            try:
                sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
                onnx_used = True
            except Exception as e_cpu:
                print(f"{timestamp_ms()} | ONNX CPU failed: {e_cpu}")
                sess = None
    except Exception as e_import:
        print(f"{timestamp_ms()} | onnxruntime import failed: {e_import}")
        sess = None

# If ONNX unavailable try PyTorch
if sess is None:
    try:
        import torch
        try:
            from hft_dl_engine import TinyConvNet, SEQ_LEN as _SEQ, FEATURES as _FEAT, DEVICE as _DEV
            print(f"{timestamp_ms()} | Imported TinyConvNet from hft_dl_engine")
        except Exception:
            from AITradingBot import TinyConvNet, SEQ_LEN as _SEQ, FEATURES as _FEAT, DEVICE as _DEV
            print(f"{timestamp_ms()} | Imported TinyConvNet from AITradingBot")
        pt_device = "cuda" if (torch.cuda.is_available() and not FORCE_CPU_ONLY) else "cpu"
        pt_model = TinyConvNet(SEQ_LEN, FEATURES).to(pt_device)
        pt_model.load_state_dict(torch.load(PT_WEIGHTS, map_location=pt_device))
        pt_model.eval()
        pt_used = True
        print(f"{timestamp_ms()} | PyTorch model loaded on {pt_device}")
    except Exception as e_pt:
        print(f"{timestamp_ms()} | PyTorch backend failed: {e_pt}")
        raise RuntimeError("No usable inference backend available.")

# ---------------- Feature extraction & inference ----------------
def normalize(x: np.ndarray):
    return (x - FEATURE_MEAN) / (FEATURE_STD + 1e-9)

def extract_features(tick: dict) -> np.ndarray:
    bid1 = tick['bid1_price']
    ask1 = tick['ask1_price']
    bid1_size = tick.get('bid1_size', 0.0)
    ask1_size = tick.get('ask1_size', 0.0)
    mid = (bid1 + ask1) / 2.0
    spread = ask1 - bid1
    imbalance = bid1_size / (bid1_size + ask1_size + 1e-9) if (bid1_size + ask1_size) > 0 else 0.5
    features = np.array([
        bid1, ask1, bid1_size, ask1_size,
        spread, mid, imbalance,
        tick.get('trade_flow', 0.0),
        tick.get('vwap', mid),
        tick.get('bid2_price', bid1),
        tick.get('ask2_price', ask1),
        tick.get('extra_feature', 0.0)
    ], dtype=np.float32)
    return features

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

# ---------------- Async pipeline ----------------
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
                    reg_pred, clf_pred, logits = await loop.run_in_executor(executor, onnx_infer, sess, X_input)
                else:
                    reg_pred, clf_pred, logits = await loop.run_in_executor(executor, pytorch_infer, pt_model, X_input)
                latency_ms = (time.perf_counter() - t0) * 1000
                PROM_INFER_LATENCY.observe(latency_ms)
                if reg_pred > REG_THRESHOLD and clf_pred == 2:
                    signal = "BUY"
                elif reg_pred < -REG_THRESHOLD and clf_pred == 0:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                print(f"{timestamp_ms()} | reg={reg_pred:.6f} clf={['DOWN','NEUTRAL','UP'][clf_pred]:7s} -> {signal}")
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
    # start logging and prometheus
    start_log_worker()
    q = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

    # Create broker and subscribe
    global broker
    broker = get_broker(BROKER_NAME, symbol=BROKER_SYMBOL, poll_ms=BROKER_POLL_MS)

    # connect broker (some brokers need initialization on main thread)
    try:
        broker.connect()
    except Exception as e:
        print(f"{timestamp_ms()} | Broker connect failed: {e}")
        raise

    # create callback that puts ticks into asyncio queue
    cb = tick_to_queue_callback(q)
    # subscribe ticks (this starts a background thread inside the broker)
    broker.subscribe_ticks(BROKER_SYMBOL, cb)

    producer = None  # we use broker's background thread as producer
    consumer = asyncio.create_task(tick_consumer(q))

    try:
        await consumer  # run until cancelled
    except asyncio.CancelledError:
        pass
    finally:
        try:
            broker.close()
        except Exception:
            pass

if __name__ == "__main__":
    print(f"{timestamp_ms()} | Starting hybrid engine. Broker={BROKER_NAME} symbol={BROKER_SYMBOL}")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("CTRL+C detected. Shutting down")
    finally:
        print("Flushing logs and stopping logger")
        flush_sync()
        print("Shutdown complete.")
