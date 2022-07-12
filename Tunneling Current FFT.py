# -*- coding: utf-8 -*-
"""
Dejia Kong
7/12/2022

Fast Fourier Transform

https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
"""
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk     # from tkinter import Tk for Python 3.x

from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
Tk().wm_attributes('-topmost', 1)# force GUI opened on top
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)

fin=open(filename)

t=[]
x=[]
fin.readline()
fin.readline()
for line in fin:
    data=line.rstrip("\r\n").split(",")
    t.append(float(data[0]))
    x.append(float(data[2]))

plt.style.use('seaborn-poster')
    

sr = 43.4 
#sampling rate
ts = 1.0/sr 
# sampling interval
from numpy.fft import fft, ifft

X = fft(x)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

plt.figure(figsize = (12, 6))
plt.subplot(121)

plt.stem(freq, np.abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 100)
plt.ylim(0, 0.00000001)
#scaling y-axis

plt.subplot(122)
plt.plot(t, ifft(X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.tight_layout()
plt.show()