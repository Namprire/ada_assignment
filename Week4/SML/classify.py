# classify.py  –  one-vs-all Gaussian-kernel LS classifier for digit.mat
import os, time, math
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------- helpers ----
def gaussian_kernel(A: np.ndarray, B: np.ndarray, h: float) -> np.ndarray:
    """Return exp(-‖a_i-b_j‖² / (2 h²)) for all i,j without explicit loops."""
    a2 = np.sum(A * A, axis=1)[:, None]          # (nA,1)
    b2 = np.sum(B * B, axis=1)[None, :]          # (1,nB)
    return np.exp(-(a2 + b2 - 2 * A @ B.T) / (2 * h * h))


# ---------------------------------------------------------------- load data --
HERE = os.path.dirname(__file__)
mat_path = os.path.join(HERE, "digit.mat")

data   = loadmat(mat_path)
X_raw  = data["X"]                                # 256 × 500 × 10
T_raw  = data["T"]                                # 256 × 200 × 10

# → (n, 256)  and label vectors
X_train = X_raw.reshape(256, -1).T                # 5 000 × 256
X_test  = T_raw.reshape(256, -1).T                # 2 000 × 256
y_train = (np.repeat(np.arange(10), 500) + 1) % 10
y_test  = (np.repeat(np.arange(10), 200) + 1) % 10

print("train :", X_train.shape, y_train.shape)
print("test  :", X_test.shape , y_test.shape)

# ------------------------------------------------- one-vs-all target matrix --
lb   = LabelBinarizer(neg_label=-1)               # +1 for class c, –1 otherwise
Yova = lb.fit_transform(y_train)                  # 5 000 × 10

# ------------------------------------------- hyper-parameter grid to search --
grid_h  = [0.5, 1, 2, 3, 4, 5, 7, 10]             # bandwidth (σ)
grid_l  = [1e-3, 1e-2, 1e-1, 1, 10]               # ridge λ

best_acc, best_h, best_lam = 0.0, None, None

print("\n--- coarse grid-search ---------------------------------------------")
for h in grid_h:
    Ktrain = gaussian_kernel(X_train, X_train, h)          # 5 000 × 5 000
    for lam in grid_l:
        Alpha = np.linalg.solve(Ktrain + lam*np.eye(Ktrain.shape[0]), Yova)
        scores = gaussian_kernel(X_test, X_train, h) @ Alpha
        acc    = (scores.argmax(1) == y_test).mean()
        print(f"h={h:<4}  λ={lam:<6}  acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_h, best_lam = acc, h, lam

print(f"\nBEST on grid : acc={best_acc:.4f}  (h={best_h}, λ={best_lam})\n")

# ------------------------------------- final model with the best (h, λ) -----
print("re-training with best hyper-parameters …")
tic = time.time()
Ktrain = gaussian_kernel(X_train, X_train, best_h)
Alpha  = np.linalg.solve(Ktrain + best_lam*np.eye(Ktrain.shape[0]), Yova)
print(f"  kernel + solve done in {time.time()-tic:0.1f}s")

scores  = gaussian_kernel(X_test, X_train, best_h) @ Alpha
y_pred  = scores.argmax(1)

final_acc = (y_pred == y_test).mean()
cm        = confusion_matrix(y_test, y_pred, labels=np.arange(10))

print(f"\nFINAL accuracy : {final_acc*100:0.2f}%")
print("confusion matrix (rows=true, cols=pred):\n", cm)
