#!/usr/bin/env python3

import matplotlib as mpl
import numpy as np
import sys

sample_rate = 1.041666e6

# Utilization here is bip_plot_csv file_name_with_data.csv

mpl.rcParams['agg.path.chunksize'] = 1000000000

import matplotlib.pyplot as plt

# with open(sys.argv[1]) as f:
#     signal = [float(i) for i in f]
# with open("transitions_guessed_delta.csv") as f:
#     delta = [float(i) for i in f]
# with open("transitions_guessed_mean.csv") as f:
#     mean = [float(i) for i in f]
# with open("transitions_guessed_canny.csv") as f:
#     canny = [float(i) for i in f]

y_signal_raw = np.loadtxt(open(sys.argv[1]))
y_mean =  np.loadtxt(open("transitions_guessed_mean.csv"))
y_delta = np.loadtxt(open("transitions_guessed_delta.csv"))
y_canny = np.loadtxt(open("transitions_guessed_canny.csv"))

points_sig = int(y_signal_raw[0])
points_result = len(y_mean)
print(points_sig)
print(points_result)
y_signal = np.delete(y_signal_raw, 0)

x_axis = np.empty(points_sig)
for i in range(points_sig):
    x_axis[i] = i/sample_rate

x_axis_res = np.empty(points_result)
for i in range(points_result):
    x_axis_res[i] = i/sample_rate

# y_signal = np.array(signal)
# y_mean = np.array(mean)
# y_delta = np.array(delta)
# y_canny = np.array(canny)
x_sig = np.array(x_axis)
x_result = np.array(x_axis_res)

f, ax = plt.subplots(figsize=(8,6))
plt.xlim(0.9, 1.105)
plt.ylim(10, 18)
plt.plot(x_sig, y_signal, color='blue', linewidth=1, alpha=0.25, label='signal')
plt.plot(x_result, y_mean, color='red', linewidth=0, alpha=1, label='mean', marker='.')
plt.plot(x_result, y_delta, color='green', linewidth=0, alpha=1, label='delta', marker='x')
plt.plot(x_result, y_canny, color='orange', linewidth=0, alpha=1, label='canny', marker='+')

plt.legend(loc='lower left')
f.tight_layout()
plt.show()
