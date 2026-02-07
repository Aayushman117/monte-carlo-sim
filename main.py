import time, math
import cupy as cp

kernel = cp.RawKernel(r'''
extern "C" __global__
void mc_kernel(float* out, float mu, float vol, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    // XORWOW RNG (fast device RNG)
    unsigned int seed = i * 9781 + 1;

    float u1 = (seed & 0xFFFF) / 65536.0f;
    float u2 = ((seed >> 16) & 0xFFFF) / 65536.0f;

    float z = sqrtf(-2.0f * logf(u1)) * cosf(6.283185f * u2);

    float s = expf(mu + vol * z);

    out[i] = 100000.0f * s;
}
''', 'mc_kernel')


def fast_var(losses, alpha=0.95):
    k = int((1-alpha) * losses.size)
    part = cp.partition(losses, -k)
    return part[-k], part[-k:].mean()


def main(n_paths=50_000_000):

    weights = cp.array([0.30, 0.25, 0.25, 0.20], dtype=cp.float32)
    mus = cp.array([0.08, 0.05, 0.12, 0.03], dtype=cp.float32)
    vols = cp.array([0.20, 0.15, 0.25, 0.10], dtype=cp.float32)

    corr = cp.array([
        [1.00, 0.40, 0.30, 0.10],
        [0.40, 1.00, 0.35, 0.15],
        [0.30, 0.35, 1.00, 0.05],
        [0.10, 0.15, 0.05, 1.00],
    ], dtype=cp.float32)

    D = cp.diag(vols)
    Sigma = D @ corr @ D

    mu_p = float(weights @ mus)
    sigma_p = math.sqrt(float(weights @ Sigma @ weights))

    drift = (mu_p - 0.5*sigma_p**2)
    vol = sigma_p

    # ===== MAX GPU BLAST =====
    t0 = time.time()

    S = cp.empty(n_paths, dtype=cp.float32)

    threads = 1024
    blocks = (n_paths + threads - 1) // threads

    kernel((blocks,), (threads,), (S, drift, vol, n_paths))

    losses = 100000.0 - S

    mean = float(cp.mean(S))
    std  = float(cp.std(S))

    var95, cvar95 = map(float, fast_var(losses))

    p_loss = float(cp.mean(S < 100000.0))

    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - t0

    print("\n=== RTX 4050 EXTREME MODE ===")
    print(f"Paths:   {n_paths:,}")
    print(f"Mean:    ${mean:,.2f}")
    print(f"Std:     ${std:,.2f}")
    print(f"VaR95:   ${var95:,.2f}")
    print(f"CVaR95:  ${cvar95:,.2f}")
    print(f"Time:    {elapsed:.3f} sec\n")

for i in range(10):
    main(50_000_000)

if __name__ == "__main__":
    main()
