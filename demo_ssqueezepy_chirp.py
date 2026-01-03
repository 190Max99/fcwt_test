import numpy as np
import matplotlib.pyplot as plt

from ssqueezepy import ssq_cwt, cwt, Wavelet, imshow
from ssqueezepy.experimental import scale_to_freq

# ----------------------------
# 1) 生成 chirp（保持你原始写法风格）
# ----------------------------
fs = 1000
n = fs * 100
ts = np.arange(n, dtype=np.float64)
t = ts / fs

f0, f1 = 1, 100
A = (f1 - f0) / 2
x = np.sin(2*np.pi * ((f0 + (A*ts)/n) * (ts/fs)))

# ----------------------------
# 2) CWT
# ----------------------------
wavelet = Wavelet()  # 默认小波
Wx, scales = cwt(x, wavelet=wavelet)

# scales -> Hz（注意：这个 freq 轴通常是非均匀的）
freqs_cwt = scale_to_freq(scales, wavelet, N=len(x), fs=fs)

# ----------------------------
# 3) SSQ-CWT
# ----------------------------
Tx, Wx2, ssq_freqs, scales2, *_ = ssq_cwt(x, wavelet=wavelet)

# ssq_freqs 有时可能是 cycles/sample，这里自适应转 Hz
ssq_freqs = np.asarray(ssq_freqs)
if ssq_freqs.max() <= 1.0:
    ssq_freqs = ssq_freqs * fs

# ----------------------------
# 4) 画图（正确时间轴 + 正确频率轴）
# ----------------------------
plt.figure(figsize=(10, 2.5))
plt.plot(t, x)
plt.title("Chirp signal")
plt.xlabel("Time (s)")
plt.tight_layout()

# 只显示关心频段
band_cwt = (freqs_cwt >= f0) & (freqs_cwt <= f1)
band_ssq = (ssq_freqs >= f0) & (ssq_freqs <= f1)

# ssqueezepy 自带 imshow：用 yticks 传入真实频率轴，避免 extent 拉歪
ikw = dict(abs=1, xticks=t, xlabel="Time (s)", ylabel="Frequency (Hz)")

imshow(Wx[band_cwt], **ikw, yticks=freqs_cwt[band_cwt], title="CWT (ssqueezepy)")
imshow(Tx[band_ssq], **ikw, yticks=ssq_freqs[band_ssq], title="SSQ-CWT (ssqueezepy)")

plt.show()
