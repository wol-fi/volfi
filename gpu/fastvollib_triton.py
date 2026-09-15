# fastvollib_triton.py -- fast-vollib's fused Triton Jaeckel kernel (arXiv:2604.27210) on the H100, on OUR
# feed, the same convention as our [5b]/[5c] lines: resident batch, CUDA events, and once with uploads+readback.
# Run in the CUDA container after:  pip install fast-vollib torch triton   (needs the licensed feed only as h,c)
#   python3 fastvollib_triton.py market_feed_ref.txt
import sys, time, numpy as np, torch, fast_vollib
from fast_vollib.jackel.triton_kernels import jackel_iv_triton
fn = sys.argv[1] if len(sys.argv) > 1 else "market_feed_ref.txt"
D = np.loadtxt(fn); h, c, vref, reg = D[:, 0], D[:, 1], D[:, 2], D[:, 3].astype(int)
TILE = int(sys.argv[2]) if len(sys.argv) > 2 else 166   # 30,000 x 166 = 4,980,000 ~ the 5-million tile of [5b]
dev = torch.device("cuda"); print(torch.cuda.get_device_name(0), "fast-vollib", fast_vollib.__version__, "torch", torch.__version__)
def to_t(a): return torch.tensor(a, dtype=torch.float64, device=dev)
c1, K1 = to_t(c), to_t(np.exp(h))
sig = jackel_iv_triton(c1, 1.0, K1, 1.0, True).cpu().numpy()   # F=1, t=1: sigma = sqrt(v)
rel = np.abs(sig - vref) / vref
print(f"accuracy on feed (n={len(h)}): max rel {np.nanmax(rel):.2e} at h={h[np.nanargmax(rel)]:.3e} c={c[np.nanargmax(rel)]:.3e}  mean {np.nanmean(rel):.2e}  nonfinite {(~np.isfinite(sig)).sum()}")
for k, nm in enumerate(["Wing", "Near", "Far", "Upper"]):
    m = reg == k
    if m.any(): print(f"   {nm:6s} n={m.sum():6d} max {np.nanmax(rel[m]):.2e} mean {np.nanmean(rel[m]):.2e}")
# timing: file order, tiled
cN, KN = c1.repeat(TILE), K1.repeat(TILE); N = cN.shape[0]
for _ in range(3): jackel_iv_triton(cN, 1.0, KN, 1.0, True)
torch.cuda.synchronize()
def ev(f, reps=7):
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1e6 / N)
    ts.sort(); return ts[len(ts) // 2], ts[0]
m, b = ev(lambda: jackel_iv_triton(cN, 1.0, KN, 1.0, True))
print(f"[resident, file order]   n={N}  median {m:.4f} ns/quote  best {b:.4f}")
idx = torch.argsort(KN); cS, KS = cN[idx].contiguous(), KN[idx].contiguous()
m, b = ev(lambda: jackel_iv_triton(cS, 1.0, KS, 1.0, True))
print(f"[resident, sorted by h]  n={N}  median {m:.4f} ns/quote  best {b:.4f}")
cH, KH = np.tile(c, TILE), np.tile(np.exp(h), TILE)
def full():
    a, k = torch.from_numpy(cH).to(dev), torch.from_numpy(KH).to(dev)
    return jackel_iv_triton(a, 1.0, k, 1.0, True).cpu()
ts = []
for _ in range(7):
    torch.cuda.synchronize(); t0 = time.perf_counter(); full(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e9 / N)
ts.sort(); print(f"[with uploads+readback]  n={N}  median {ts[3]:.4f} ns/quote  best {ts[0]:.4f}")
