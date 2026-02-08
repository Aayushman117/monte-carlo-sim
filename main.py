import cupy as cp
import time
from metrics import start_gpu_monitor, plot_gpu_log

# ============================================================
# CONFIG
# ============================================================

TOTAL_PATHS = 100_000_000
BATCH_PATHS = 2_000_000
BATCHES = TOTAL_PATHS // BATCH_PATHS

ASSETS = 10                     # 🔥 change to 20, 30, 50
STEPS = 252

S0 = cp.array([100.0] * ASSETS, dtype=cp.float32)
MU = cp.array([0.07] * ASSETS, dtype=cp.float32)
SIGMA = cp.array([0.25] * ASSETS, dtype=cp.float32)

T = 1.0
DT = T / STEPS

# Simple equal-weight portfolio
WEIGHTS = cp.ones(ASSETS, dtype=cp.float32) / ASSETS
STRIKE = 100.0

PATH_DTYPE = cp.float32
ACC_DTYPE = cp.float64

cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

# ============================================================
# CORRELATION (REALISTIC)
# ============================================================

rho = 0.4
corr = rho * cp.ones((ASSETS, ASSETS), dtype=cp.float32)
cp.fill_diagonal(corr, 1.0)

L = cp.linalg.cholesky(corr)

# ============================================================
# BUFFERS
# ============================================================

Z = cp.empty((BATCH_PATHS, ASSETS), dtype=PATH_DTYPE)
Z_anti = cp.empty_like(Z)

S = cp.empty_like(Z)
S_anti = cp.empty_like(Z)

S_sum = cp.zeros_like(Z)
S_sum_anti = cp.zeros_like(Z)

# ============================================================
# MONITOR
# ============================================================

stop_event, gpu_log, monitor_thread = start_gpu_monitor()

print("\nRunning correlated multi-asset Monte Carlo...")
t0 = time.time()

# Welford stats
count = cp.float64(0.0)
mean = cp.float64(0.0)
m2 = cp.float64(0.0)

# Store losses for VaR / CVaR (streaming reservoir)
losses = []

drift = (MU - 0.5 * SIGMA**2) * DT
vol = SIGMA * cp.sqrt(DT)

# ============================================================
# MAIN LOOP
# ============================================================

for b in range(BATCHES):
    print(f"Batch {b+1}/{BATCHES}")

    S[:] = S0
    S_anti[:] = S0
    S_sum.fill(0.0)
    S_sum_anti.fill(0.0)

    for _ in range(STEPS):
        Z[:] = cp.random.standard_normal(Z.shape, dtype=PATH_DTYPE)
        Z_anti[:] = -Z

        Zc = Z @ L.T
        Zc_anti = Z_anti @ L.T

        S *= cp.exp(drift + vol * Zc)
        S_anti *= cp.exp(drift + vol * Zc_anti)

        S_sum += S
        S_sum_anti += S_anti

    # Portfolio average price
    avg = (S_sum / STEPS) @ WEIGHTS
    avg_anti = (S_sum_anti / STEPS) @ WEIGHTS

    payoff = 0.5 * (
        cp.maximum(avg - STRIKE, 0.0) +
        cp.maximum(avg_anti - STRIKE, 0.0)
    )

    # Stats
    batch_mean = cp.mean(payoff, dtype=ACC_DTYPE)
    batch_var = cp.var(payoff, dtype=ACC_DTYPE)

    n = cp.float64(BATCH_PATHS)
    delta = batch_mean - mean
    new_count = count + n

    mean += delta * n / new_count
    m2 += batch_var * n + delta**2 * count * n / new_count
    count = new_count

    # Store subset for VaR / CVaR
    losses.append(-payoff[:100_000].get())

cp.cuda.Stream.null.synchronize()
elapsed = time.time() - t0

stop_event.set()
monitor_thread.join()

std = cp.sqrt(m2 / count)

# ============================================================
# VaR / CVaR
# ============================================================

import numpy as np

losses = np.concatenate(losses)
alpha = 0.99

VaR = np.quantile(losses, alpha)
CVaR = losses[losses >= VaR].mean()

# ============================================================
# RESULTS
# ============================================================

print("\n=== MULTI-ASSET MONTE CARLO DONE ===")
print(f"Assets:              {ASSETS}")
print(f"Paths simulated:     {int(count):,}")
print(f"Mean payoff:         {float(mean):.6f}")
print(f"Std deviation:       {float(std):.6f}")
print(f"VaR {int(alpha*100)}%:              {VaR:.6f}")
print(f"CVaR {int(alpha*100)}%:             {CVaR:.6f}")
print(f"Elapsed time:        {elapsed:.2f} sec")

plot_gpu_log(gpu_log)
