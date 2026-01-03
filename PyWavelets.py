import numpy as np
import matplotlib.pyplot as plt
import pywt

# ----------------------------
# 1) 生成线性 chirp（和你之前 fcwt demo 同类）
# ----------------------------
fs = 1000               # 采样率 Hz
duration = 10           # 秒：先用 10s 演示（100s + 200个频点会很慢）
t = np.arange(0, duration, 1/fs)
T = t[-1] if len(t) > 1 else duration

f0 = 1                  # 起始频率 Hz
f1 = 101                # 终止频率 Hz
k = (f1 - f0) / T        # 线性扫频斜率 Hz/s

# 相位：phi(t)=2π( f0*t + 0.5*k*t^2 )，瞬时频率 f(t)=f0+k*t
phi = 2*np.pi*(f0*t + 0.5*k*t**2)
signal = np.sin(phi)

# ----------------------------
# 2) 用 PyWavelets 做 CWT
# ----------------------------
wavelet = "cmor1.5-1.0"  # 文档里常用的复 Morlet 配置 :contentReference[oaicite:1]{index=1}

fn = 200
freqs_target = np.linspace(f0, f1, fn)     # 你希望看的频率轴（Hz）
dt = 1 / fs

# 频率 <-> 尺度：freq ≈ central_frequency / (scale * dt)
# => scale = central_frequency / (freq * dt)
cf = pywt.central_frequency(wavelet)
scales = cf / (freqs_target * dt)

# method='fft' 通常比 conv 快（尤其长信号）:contentReference[oaicite:2]{index=2}
cwt_coef, freqs_out = pywt.cwt(
    signal,
    scales,
    wavelet,
    sampling_period=dt,
    method="fft"
)

power = np.abs(cwt_coef)  # 复小波输出是复数，可视化一般取幅值 :contentReference[oaicite:3]{index=3}

# ----------------------------
# 3) 画图：信号 + 时频图
# ----------------------------
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

axs[0].plot(t, signal)
axs[0].set_ylabel("Amplitude")
axs[0].set_title("Chirp signal")

pcm = axs[1].pcolormesh(t, freqs_out, power, shading="auto")
axs[1].set_ylabel("Frequency (Hz)")
axs[1].set_xlabel("Time (s)")
axs[1].set_title("PyWavelets CWT (Scaleogram)")
fig.colorbar(pcm, ax=axs[1])

plt.tight_layout()
plt.show()
