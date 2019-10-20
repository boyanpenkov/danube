import sys
import numpy as np
import struct
import gc

filename = sys.argv[1]

signal = []

with open(filename) as f:
        for line in f:
	        signal.append(float(line))

filename = filename[:-3]
filename = filename + "dat"
newFile = open(filename, "wb")

myarray = np.asarray(signal)
del signal
gc.collect()

format = len(myarray)
format = str(format) + "f"
newFile.write(struct.pack(format, *myarray))
newFileByteArray = bytearray(myarray)
