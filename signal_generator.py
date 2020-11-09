import numpy as np
import sys

import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt

points = int(sys.argv[1]) # total points
frequency = 50000 # fake made-up sample frequency of input signal -- Chimera is 1MHz

high_mean = 10
current_sd = 0.1

event_space = 1 # second
event_time = 0.1 # second
event_time_sd = 0.05 # second
event_depth = 0.6
event_depth_sd = 0.05

print "I see a point count of "+str(points)
print "I see a frequency of "+str(frequency)
print "I calculate EVENTS = "+str(points/frequency)
print "NOW GO EDIT THE VALUE in find_events.cu -- and yes, TODO a better way of handing this"

mu, sigma = high_mean, current_sd
s = np.float16(np.random.normal(mu, sigma, points))

x_axis = np.arange(points, dtype='int32')

event_count = (points/frequency)/event_space
for start in range(1, event_count):
        go = start*points/event_count
        cur_time = np.random.normal(event_time, event_time_sd)
        cur_depth = np.random.normal(event_depth, event_depth_sd)
        for index in range(go, go+int(cur_time*frequency)):
                s[index] = s[index]*cur_depth
        
y = np.array(s)
x = np.array(x_axis)
plt.plot(x, y,
         color='blue',
         linestyle='solid',
         linewidth=2, marker='o',
         markerfacecolor='blue',
         markersize=2)

plt.savefig('generated_signal.png')
plt.close()

file_signal = open("signal.csv", "w")
print >> file_signal, points
for value in s:
        print >> file_signal, value
