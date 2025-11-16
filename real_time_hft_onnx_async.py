# real_time_hft_onnx_async.py  (fallback to PyTorch GPU if ONNX CUDA unavailable)
import asyncio
import numpy as np
import time
from collections import deque
import concurrent.futures
from datetime import datetime

# Try ONNX first
USE_ONNX = True
ONNX_PATH = "hft_model.onnx"

sess = None
try:
    import onnxruntime as ort
    try:
        # try to create CUDA session provider first
        sess = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        print("ONNX session providers:", sess.get_providers())
    except Exception as e:
        print("Failed to create ONNX session with CUDA provider, falling back to CPU or PyTorch. Error:", e)
        # try CPU ONNX (might still work)
        try:
            sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
            print("ONNX CPU session created.")
        except Exception as e2:
            print("ONNX session creation failed entirely:", e2)
            sess = None
except Exception as e:
    print("onnxruntime import failed:", e)
    sess = None

# If ONNX failed, we will use PyTorch as a fallback for GPU inference
use_pytorch = False
if sess is None:
    try:
        import torch
        from AITradingBot import TinyConvNet, SEQ_LEN, FEATURES, DEVICE  # adjust import name to your training file
        # load model weights
        pt_model = TinyConvNet(SEQ_LEN, FEATURES).to(DEVICE)
        pt_model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE, weights_only=True))
        pt_model.eval()
        use_pytorch = True
        print("Falling back to PyTorch GPU inference.")
    except Exception as e:
        print("PyTorch fallback failed:", e)
        raise RuntimeError("No usable inference backend (ONNX failed and PyTorch fallback failed).")

# config
SEQ_LEN = 64
FEATURES = 12
REG_THRESHOLD = 0.0005
clf_labels = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

tick_buffer = deque(maxlen=SEQ_LEN)
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

async def tick_producer(queue: asyncio.Queue, rate_hz=2000):
    interval = 1.0 / rate_hz
    while True:
        tick = {
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
        await queue.put(tick)
        await asyncio.sleep(interval)

# ONNX inference wrapper
def onnx_infer(session, X_input: np.ndarray):
    out = session.run(None, {"X": X_input})
    reg_pred = float(out[0].reshape(-1)[0])
    clf_logits = out[1]
    clf_pred = int(np.argmax(clf_logits, axis=1)[0])
    return reg_pred, clf_pred

# PyTorch inference wrapper (runs on GPU if DEVICE is cuda)
def pytorch_infer(model, X_input_np: np.ndarray):
    import torch
    x = torch.from_numpy(X_input_np).to(next(model.parameters()).device)
    with torch.no_grad():
        reg, clf_logits = model(x)
        reg_val = float(reg.cpu().numpy().reshape(-1)[0])
        clf_val = int(torch.argmax(clf_logits, dim=1).cpu().numpy()[0])
    return reg_val, clf_val

async def tick_consumer(queue: asyncio.Queue):
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    try:
        while True:
            tick = await queue.get()
            f = extract_features(tick)
            f = normalize(f)
            tick_buffer.append(f)

            if len(tick_buffer) == SEQ_LEN:
                X_input = np.array(tick_buffer, dtype=np.float32)[None, :, :]
                if sess is not None:
                    # ONNX path (runs in threadpool)
                    reg_pred, clf_pred = await loop.run_in_executor(executor, onnx_infer, sess, X_input)
                else:
                    # PyTorch path (run in executor too)
                    reg_pred, clf_pred = await loop.run_in_executor(executor, pytorch_infer, pt_model, X_input)

                # generate signal
                if reg_pred > REG_THRESHOLD and clf_pred == 2:
                    signal = "BUY"
                elif reg_pred < -REG_THRESHOLD and clf_pred == 0:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"{ts} | reg={reg_pred:.6f} clf={clf_labels[clf_pred]:7s} -> {signal}")
            queue.task_done()
    except asyncio.CancelledError:
        executor.shutdown(wait=False)
        raise

async def main():
    q = asyncio.Queue(maxsize=20000)
    producer = asyncio.create_task(tick_producer(q, rate_hz=1000))
    consumer = asyncio.create_task(tick_consumer(q))
    try:
        await asyncio.gather(producer, consumer)
    except KeyboardInterrupt:
        producer.cancel()
        consumer.cancel()
        await asyncio.gather(producer, consumer, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())