import numpy as np
import os

# Configuration
N_SAMPLES = 5000   # number of samples
SEQ_LEN = 64       # ticks per sample
FEATURES = 12      # features per tick

# Synthetic X: random floats simulating features
X = np.random.randn(N_SAMPLES, SEQ_LEN, FEATURES).astype(np.float32)

# Synthetic regression target: small random returns
y_reg = np.random.randn(N_SAMPLES).astype(np.float32) * 0.001

# Synthetic classification target: 0,1,2
y_clf = np.random.randint(0, 3, size=(N_SAMPLES,)).astype(np.int64)

# Save files
np.save("X.npy", X)
np.save("y_reg.npy", y_reg)
np.save("y_clf.npy", y_clf)

print("Synthetic dataset generated: X.npy, y_reg.npy, y_clf.npy")