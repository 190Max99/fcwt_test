import numpy as np
import matplotlib.pyplot as plt
import pywt

fs = 1000
n = fs * 100
ts = np.arange(n)
t = ts / fs

f0, f1 = 1, 100
A = (f1 - f0) / 2
signal = np.sin(2*np.pi * ((f0 + (A*ts)/n) * (ts/fs)))

wavelet = "cmor1.5-1.0"
fn = 200
freqs_target = np.linspace(f0, f1, fn)
dt = 1 / fs

cf = pywt.central_frequency(wavelet)
scales = cf / (freqs_target * dt)

cwt_coef, freqs_out = pywt.cwt(signal, scales, wavelet,
                               sampling_period=dt, method="fft")
power = np.abs(cwt_coef)

# ===== 画图：不要用 pcolormesh 画 2000万格子，改用 imshow =====
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

axs[0].plot(t, signal)
axs[0].set_ylabel("Amplitude")
axs[0].set_title("Chirp signal")

im = axs[1].imshow(
    power,
    aspect="auto",
    origin="lower",
    extent=[t[0], t[-1], freqs_out[0], freqs_out[-1]]
)
axs[1].set_ylabel("Frequency (Hz)")
axs[1].set_xlabel("Time (s)")
axs[1].set_title("PyWavelets CWT (Scaleogram)")
fig.colorbar(im, ax=axs[1])

plt.tight_layout()
plt.show()
