import os
# 固定线程，避免 BLAS/FFT/OMP 偷开多线程导致结果飘
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import time
import numpy as np

# ------- 你的算法库 -------
import fcwt
import ccwt

def make_linear_chirp(fs=1000, T=100, f_start=1.0, f_end=41.0):
    """标准线性chirp：f(t)=f_start + (f_end-f_start)*t/T"""
    n = fs * T
    t = np.arange(n, dtype=np.float64) / fs
    k = (f_end - f_start) / T
    x = np.sin(2*np.pi*(f_start*t + 0.5*k*t*t)).astype(np.float32)
    return x, n

def run_fcwt(x, fs, fmin, fmax, fn):
    freqs, out = fcwt.cwt(x, fs, fmin, fmax, fn)
    return out

def run_ccwt(x, fs, n, fmin, fmax, fn, padding=0, thread_count=1, deviation=0.7):
    """
    CCWT 频率单位是“归一化频率”u = f(Hz) * width / fs
    为了和 fCWT 对齐，把 width 设成 n
    """
    width = n
    height = fn

    u0 = fmin * width / fs
    u1 = fmax * width / fs
    frequency_offset = u0
    frequency_range = (u1 - u0)

    frequency_band = ccwt.frequency_band(
        height, frequency_range, frequency_offset, 0.0, deviation
    )

    F = ccwt.fft(x, padding, thread_count)
    out = ccwt.numeric_output(F, frequency_band, width, padding, thread_count)
    return out

def bench(name, func, warmup=1, repeats=5):
    # warm-up（预热一次，避免首次 cache/plan 影响）
    for _ in range(warmup):
        _ = func()
    gc.collect()

    wall = []
    cpu = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        c0 = time.process_time()
        _ = func()
        c1 = time.process_time()
        t1 = time.perf_counter()
        wall.append(t1 - t0)
        cpu.append(c1 - c0)
        gc.collect()

    wall = np.array(wall)
    cpu = np.array(cpu)
    print(f"\n[{name}]")
    print(f"Wall avg={wall.mean():.4f}s  min={wall.min():.4f}s  std={wall.std():.4f}s")
    print(f"CPU  avg={cpu.mean():.4f}s  min={cpu.min():.4f}s")

def main():
    fs = 1000
    T = 100

    # ① 生成同一个 chirp
    # 你之前那种写法最终到 ~41Hz，我这里默认也给 41Hz，方便对齐你现在图
    x, n = make_linear_chirp(fs=fs, T=T, f_start=1.0, f_end=41.0)

    # ② 分析频率范围（对齐两算法）
    fmin, fmax, fn = 1.0, 101.0, 200

    # ③ CCWT 参数
    padding = 0
    thread_count = 1
    deviation = 0.7

    bench("fCWT", lambda: run_fcwt(x, fs, fmin, fmax, fn), warmup=1, repeats=5)
    bench("CCWT", lambda: run_ccwt(x, fs, n, fmin, fmax, fn,
                                   padding=padding, thread_count=thread_count, deviation=deviation),
          warmup=1, repeats=5)

if __name__ == "__main__":
    main()
