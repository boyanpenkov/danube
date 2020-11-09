import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys

mpl.rcParams['agg.path.chunksize'] = 10000

if (len(sys.argv) != 1):
    method = str(sys.argv[1])

    if (method == "one"):

        with open(sys.argv[2]) as f:
            signal = map(float, f)

        points = int(signal[0])

        signal.pop(0)

        x_axis = []

        for i in range(points):
            x_axis.append(i)

        y_signal = np.array(signal)
        x = np.array(x_axis)

        plt.plot(x, y_signal, color='blue', linewidth=2, marker='.',
                 markerfacecolor='blue', markersize=2, alpha=0.5)
        plt.show()

    elif (method == "two"):

        with open(sys.argv[2]) as f:
            signal1 = map(float, f)
        with open(sys.argv[3]) as f:
            signal2 = map(float, f)

        points1 = int(signal1[0])
        points2 = int(signal2[0])

        signal1.pop(0)
        signal2.pop(0)

        x_axis1 = []
        for i in range(points1):
            x_axis1.append(i)
        x_axis2 = []
        for i in range(points2):
            x_axis2.append(i)

        y_signal1 = np.array(signal1)
        y_signal2 = np.array(signal2)
        x1 = np.array(x_axis1)
        x2 = np.array(x_axis2)

        plt.subplot(2, 1, 1)
        plt.plot(x1, y_signal1, color='blue', linewidth=2, marker='.',
                 markerfacecolor='blue', markersize=2, alpha=0.5)

        plt.subplot(2, 1, 2)
        plt.plot(x2, y_signal2, color='blue', linewidth=2, marker='.',
                 markerfacecolor='blue', markersize=2, alpha=0.5)

        plt.show()

    elif (method == "three"):

        with open(sys.argv[2]) as f:
            signal1 = map(float, f)
        with open(sys.argv[3]) as f:
            signal2 = map(float, f)
        with open(sys.argv[4]) as f:
            signal3 = map(float, f)

        points1 = int(signal1[0])
        points2 = int(signal2[0])
        points3 = int(signal3[0])

        signal1.pop(0)
        signal2.pop(0)
        signal3.pop(0)

        x_axis1 = []
        for i in range(points1):
            x_axis1.append(i)
        x_axis2 = []
        for i in range(points2):
            x_axis2.append(i)
        x_axis3 = []
        for i in range(points3):
            x_axis3.append(i)

        y_signal1 = np.array(signal1)
        y_signal2 = np.array(signal2)
        y_signal3 = np.array(signal3)
        x1 = np.array(x_axis1)
        x2 = np.array(x_axis2)
        x3 = np.array(x_axis3)

        plt.subplot(3, 1, 1)
        plt.plot(x1, y_signal1, color='blue', linewidth=2, marker='.',
                 markerfacecolor='blue', markersize=2, alpha=0.5)

        plt.subplot(3, 1, 2)
        plt.plot(x2, y_signal2, color='blue', linewidth=2, marker='.',
                 markerfacecolor='blue', markersize=2, alpha=0.5)

        plt.subplot(3, 1, 3)
        plt.plot(x3, y_signal3, color='blue', linewidth=2, marker='.',
                 markerfacecolor='blue', markersize=2, alpha=0.5)

        plt.show()

    elif (method == "gradient"):

        with open(sys.argv[2]) as f:
            signal = map(float, f)

        points = int(signal[0])

        signal.pop(0)

        x_axis = []

        for i in range(points):
            x_axis.append(i)

        y_signal1 = np.array(signal)
        y_signal2 = np.gradient(signal)
        x = np.array(x_axis)

        plt.subplot(2, 1, 1)
        plt.plot(x, y_signal1, color='blue', linewidth=2, marker='.',
                 markerfacecolor='blue', markersize=2, alpha=0.5)

        plt.subplot(2, 1, 2)
        plt.plot(x, y_signal2, color='blue', linewidth=2, marker='.',
                 markerfacecolor='blue', markersize=2, alpha=0.5)

        plt.show()

else:

    with open(sys.argv[1]) as f:
        signal = map(float, f)
    print signal

    with open("transitions_guessed_delta.csv") as f:
        delta = map(float, f)

    with open("transitions_guessed_mean.csv") as f:
        mean = map(float, f)

    with open("transitions_guessed_canny.csv") as f:
        canny = map(float, f)

    points = int(signal[0])

    signal.pop(0)

    x_axis = []

    for i in range(points):
        x_axis.append(i)

    y_signal = np.array(signal)
    y_mean = np.array(mean)
    y_delta = np.array(delta)
    y_canny = np.array(canny)
    x = np.array(x_axis)

    plt.plot(x, y_delta, color='green', linestyle='solid', linewidth=1,
             marker='o', markerfacecolor='green', markersize=2, alpha=0.5)
    plt.plot(x, y_mean, color='red', linestyle='solid', linewidth=1,
             marker='o', markerfacecolor='red', markersize=2, alpha=0.5)
    plt.plot(x, y_signal, color='blue', linewidth=2, marker='.',
             markerfacecolor='blue', markersize=2, alpha=0.5)
    plt.plot(x, y_canny, color='yellow', linewidth=2, marker='.',
             markerfacecolor='yellow', markersize=2, alpha=0.5)
    plt.show()
