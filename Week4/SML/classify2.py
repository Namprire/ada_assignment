#!/usr/bin/env python3
"""
Homework 2 – Least-Squares Classification with a Gaussian Kernel
reference: ADA4-LSclassification slides 56-59
"""

import os, time
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------- helpers ----
def gaussian_kernel(A: np.ndarray, B: np.ndarray, h: float) -> np.ndarray:
    """Compute exp(-‖a_i−b_j‖² / (2 h²)) for all i,j (vectorised)."""
    a2 = np.sum(A * A, axis=1)[:, None]          # (nA,1)
    b2 = np.sum(B * B, axis=1)[None, :]          # (1 ,nB)
    return np.exp(-(a2 + b2 - 2 * A @ B.T) / (2 * h * h))

# ---------------------------------------------------------------- load data --
mat_path = os.path.join(os.path.dirname(__file__), "digit.mat")
data     = loadmat(mat_path)

# MATLAB tensors  (pixels × samples × digit)
X_ten, T_ten = data["X"], data["T"]               # 256×500×10 and 256×200×10

# reshape → (n_samples, 256) row-vectors
# keep MATLAB sample order (Fortran / column-major)!
X_train = X_ten.reshape(256, -1, order="F").T      # 5 000 × 256
X_test  = T_ten.reshape(256, -1, order="F").T      # 2 000 × 256

n_train = X_train.shape[0]

# labels 0-9 in MATLAB order “1..9 then 0”
y_train = (np.repeat(np.arange(10), 500) + 1) % 10   # length 5 000
y_test  = (np.repeat(np.arange(10), 200) + 1) % 10   # length 2 000

print(f"train : {X_train.shape}   test : {X_test.shape}")

# ------------------------------------------------ hyper-params from slides --
h   = 10.0            # bandwidth  (slide uses hh = 2*10² ⇒ h=10)
lam = 1.0             # λ = 1

# --------------------------------------------------- build train kernel -----
print("\ncomputing 5 000×5 000 Gaussian kernel …", end="", flush=True)
tic = time.time()
K   = gaussian_kernel(X_train, X_train, h)
print(f" done in {time.time()-tic:0.1f}s")

# --------------------------------------------------- one-vs-all targets -----
# Y_ij = +1 if sample i is digit j else –1   (shape 5 000 × 10)
Y = np.where(y_train[:, None] == np.arange(10)[None, :], 1.0, -1.0)

# ------------------------------------------------ train (slide formula) -----
# Θ = (K² + λI)⁻¹  K  Y    ← slide 22 (green MATLAB box)
print("solving (K²+λI) Θ = K Y …", end="", flush=True)
tic = time.time()
Theta = np.linalg.solve(K @ K + lam * np.eye(n_train), K @ Y)
print(f" done in {time.time()-tic:0.1f}s")

# ---------------------------------------------------- predict test ----------
Ktest  = gaussian_kernel(X_test, X_train, h)      # 2 000 × 5 000
scores = Ktest @ Theta                            # 2 000 × 10
y_pred = scores.argmax(axis=1)

acc = (y_pred == y_test).mean()
cm  = confusion_matrix(y_test, y_pred, labels=np.arange(10))

print(f"\noverall accuracy : {acc*100:0.2f}%  (expect ≈ 95 %)")
print("confusion matrix  (rows=true, cols=pred):\n", cm)
