import fcwt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# 1) 生成你的 chirp 信号
# -----------------------------
fs = 1000
n = fs * 100  # 100 seconds
ts = np.arange(n)

signal = np.sin(2*np.pi*((1+(20*ts)/n)*(ts/fs))).astype(np.float32)

f0, f1, fn = 1, 101, 200

# -----------------------------
# 2) 计算 fCWT（一次性算完）
# -----------------------------
freqs, out = fcwt.cwt(signal, fs, f0, f1, fn)
mag = np.abs(out)  # 画幅值更直观

T = n / fs
t = ts / fs

# -----------------------------
# 3) 动画显示：逐列“揭开”时频图
# -----------------------------
fig = plt.figure(figsize=(10, 6))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# 上图：输入信号
ax1.plot(t, signal, linewidth=1)
ax1.set_title("Input signal")
ax1.set_xlim(0, T)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")

# 下图：CWT（先放一张全 0 的图）
disp = np.zeros_like(mag)
im = ax2.imshow(
    disp,
    aspect="auto",
    origin="lower",
    extent=[0, T, freqs[0], freqs[-1]]
)
ax2.set_title("fCWT (animated reveal)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Frequency (Hz)")
plt.colorbar(im, ax=ax2, label="|CWT|")

# 你可以调这个：每隔多少个采样更新一帧（越大越快也越粗糙）
step = 200
frames = range(0, n, step)

def update(k):
    # 把 0..k 的列填进去，其余保持 0（就像逐步展开）
    disp[:, :k] = mag[:, :k]
    im.set_data(disp)
    return (im,)

ani = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)

plt.tight_layout()
plt.show()
