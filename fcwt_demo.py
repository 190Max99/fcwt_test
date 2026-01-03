import fcwt
import numpy as np
import matplotlib.pyplot as plt

# #Initialize
# fs = 1000
# n = fs*100 #100 seconds
# ts = np.arange(n)

# #Generate linear chirp
# signal = np.sin(2*np.pi*((1+(20*ts)/n)*(ts/fs)))

fs = 1000
n = fs * 100
ts = np.arange(n)

f0, f1 = 1, 100
A = (f1 - f0) / 2      # 关键：A 控制扫频跨度的一半

signal = np.sin(2*np.pi * ((f0 + (A*ts)/n) * (ts/fs)))

f0 = 1 #lowest frequency
f1 = 101 #highest frequency
fn = 200 #number of frequencies

#Calculate CWT without plotting...
#freqs, out = fcwt.cwt(signal, fs, f0, f1, fn)

#... or calculate and plot CWT
fcwt.plot(signal, fs, f0=f0, f1=f1, fn=fn)