import cupy as cp
import time
from metrics import start_gpu_monitor, plot_gpu_log

N = 16384
DTYPE = cp.float32
ITER = 50

print("Allocating large matrices...")

A = cp.random.random((N, N), dtype=DTYPE)
B = cp.random.random((N, N), dtype=DTYPE)
C = cp.empty((N, N), dtype=DTYPE)

cp.cuda.Stream.null.synchronize()

print("\nMatrices allocated.")
input("Press ENTER to start max-load compute...")

# 🔥 START GPU MONITOR
stop_event, gpu_log, monitor_thread = start_gpu_monitor()

print("\nStarting MAX GPU LOAD...")
t0 = time.time()

for _ in range(ITER):
    cp.matmul(A, B, out=C)
    A, C = C, A

cp.cuda.Stream.null.synchronize()
elapsed = time.time() - t0

# 🛑 STOP MONITOR
stop_event.set()
monitor_thread.join()

print("\n=== DONE ===")
print(f"Iterations: {ITER}")
print(f"Elapsed time: {elapsed:.2f} seconds")

# 📊 SHOW GRAPH
plot_gpu_log(gpu_log)
