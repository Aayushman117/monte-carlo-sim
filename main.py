import cupy as cp
import time

# ============================================================
# CONFIG — tuned for RTX 4050 6GB
# ============================================================

N = 16384            # Matrix size
DTYPE = cp.float32   # float32 uses Tensor Cores
ITER = 50            # Increase for longer load

# ============================================================
# ALLOCATE LARGE MATRICES (~5–6 GB TOTAL)
# ============================================================

print("Allocating large matrices...")

A = cp.random.random((N, N), dtype=DTYPE)
B = cp.random.random((N, N), dtype=DTYPE)
C = cp.empty((N, N), dtype=DTYPE)

cp.cuda.Stream.null.synchronize()

print("\nMatrices allocated.")
print("👉 Check nvidia-smi NOW (VRAM should be ~5–6 GB)")
input("Press ENTER to start max-load compute...")

# ============================================================
# MAX LOAD COMPUTE LOOP
# ============================================================

print("\nStarting MAX GPU LOAD...")
t0 = time.time()

for i in range(ITER):
    # Heavy GEMM = tensor cores + bandwidth + ALUs
    cp.matmul(A, B, out=C)

    # Swap buffers to force reuse and cache pressure
    A, C = C, A

cp.cuda.Stream.null.synchronize()
elapsed = time.time() - t0

print("\n=== DONE ===")
print(f"Iterations: {ITER}")
print(f"Elapsed time: {elapsed:.2f} seconds")
