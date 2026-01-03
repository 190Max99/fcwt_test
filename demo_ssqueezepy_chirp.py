import numpy as np
import matplotlib.pyplot as plt

from ssqueezepy import ssq_cwt, cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq

# 1) 生成 chirp（和你之前 fcwt 的写法风格一致）
fs = 1000
duration = 10  # 想和你之前一样就改成 100（会更慢）
n = fs * duration
ts = np.arange(n)
t = ts / fs

x = np.sin(2*np.pi*((1 + (20*ts)/n) * (ts/fs)))  # 你之前的公式

# 2) CWT
wavelet = Wavelet()  # 默认小波
Wx, scales = cwt(x, wavelet=wavelet)
freqs_hz = scale_to_freq(scales, wavelet, N=len(x), fs=fs)

# 3) SSQ-CWT（同步挤压）
Tx, Wx2, ssq_freqs, scales2, *_ = ssq_cwt(x, wavelet=wavelet)

def plot_tf(M, freqs, title):
    M = np.abs(M)
    freqs = np.asarray(freqs)

    # 某些情况下 ssq_freqs 可能是 cycles/sample，这里自适应转 Hz
    if freqs.max() <= 1.0:
        freqs = freqs * fs

    # 排序 + 只显示 1~101 Hz（跟你之前一致）
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    M = M[idx, :]

    band = (freqs >= 1) & (freqs <= 101)
    freqs2 = freqs[band]
    M2 = M[band, :]

    plt.figure(figsize=(10, 4))
    plt.imshow(M2, aspect="auto", origin="lower",
               extent=[t[0], t[-1], freqs2[0], freqs2[-1]])
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="|coef|")
    plt.tight_layout()

plt.figure(figsize=(10, 2.5))
plt.plot(t, x)
plt.title("Chirp signal")
plt.xlabel("Time (s)")
plt.tight_layout()

plot_tf(Wx, freqs_hz, "CWT (ssqueezepy)")
plot_tf(Tx, ssq_freqs, "SSQ-CWT (ssqueezepy)")

plt.show()
