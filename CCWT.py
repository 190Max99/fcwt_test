import ccwt
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------
# 1) 生成和你一样的 chirp 信号
# -----------------------------


fs = 1000
n = fs * 100
ts = np.arange(n)

f0, f1 = 1, 100
A = (f1 - f0) / 2      # 关键：A 控制扫频跨度的一半

signal = np.sin(2*np.pi * ((f0 + (A*ts)/n) * (ts/fs)))


# f0/f1/fn：你原来的设置（单位 Hz）
f0 = 1
f1 = 101
fn = 200

# -----------------------------
# 2) CCWT 参数：height/width/padding/thread_count
# -----------------------------
height = fn              # 频率bin数量（类似你的 fn）
width  = n               # 输出时间bin数量（=n 会很大；想省内存可改小，如 4096）
padding = 0              # 可加 padding 减少首尾“环绕影响”
thread_count = 1         # 可改成 os.cpu_count() 测多线程

# -----------------------------
# 3) Hz -> CCWT 归一化频率单位
#    CCWT 的 frequency_offset / frequency_range 与 width 绑定：
#    归一化频率 u = f(Hz) * width / fs  （等价于：整段信号里有多少个周期）
# -----------------------------
u0 = f0 * width / fs
u1 = f1 * width / fs
frequency_offset = u0
frequency_range  = (u1 - u0)

# 生成频带（线性频带：frequency_basis=0.0 默认就是线性）
frequency_band = ccwt.frequency_band(
    height,
    frequency_range,
    frequency_offset
)

# -----------------------------
# 4) 计算：FFT -> CWT数值输出（不渲染PNG）
# -----------------------------
#t0 = time.perf_counter()
F = ccwt.fft(signal, padding, thread_count)  # ccwt.fft(input_signal, padding, thread_count) :contentReference[oaicite:2]{index=2}
#t1 = time.perf_counter()

#t2 = time.perf_counter()
out = ccwt.numeric_output(F, frequency_band, width, padding, thread_count)  # :contentReference[oaicite:3]{index=3}
#t3 = time.perf_counter()

# print(f"FFT time: {t1-t0:.3f} s")
# print(f"CCWT numeric_output time: {t3-t2:.3f} s")
# print("out shape:", out.shape)

# -----------------------------
# 5) 生成“类似 fCWT 返回的 freqs”用于画图（单位 Hz）
#    frequency_band[:,0] 是 CCWT 频率坐标（归一化单位），换成 Hz：
#    f_hz = u * fs / width
# -----------------------------
freqs_hz = frequency_band[:, 0] * fs / width

# -----------------------------
# 6) 可视化（幅值谱）
# -----------------------------
plt.figure(figsize=(10, 4))
# plt.imshow(
#     np.abs(out),
#     aspect="auto",
#     origin="lower",
#     extent=[0, n/fs, freqs_hz[0], freqs_hz[-1]]
# )
# plt.xlabel("Time (s)")
# plt.ylabel("Frequency (Hz)")
# plt.title("CCWT amplitude")
# plt.colorbar(label="|CWT|")
# plt.tight_layout()
# plt.show()

img = np.abs(out)
freqs_hz = frequency_band[:, 0] * fs / width  # 你之前的换算

# 如果 freqs 是降序，就反过来（并同步翻转图像行）
if freqs_hz[0] > freqs_hz[-1]:
    freqs_hz = freqs_hz[::-1]
    img = img[::-1, :]

plt.imshow(img, aspect="auto", origin="lower",
           extent=[0, n/fs, freqs_hz[0], freqs_hz[-1]])
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("CCWT amplitude")
plt.colorbar(label="|CWT|")
plt.show()
