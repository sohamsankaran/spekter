import pyaudio
import scipy
import struct
import scipy.fftpack
import threading
import time, datetime
import math
import pdb
import numpy as np
import matplotlib.pyplot as plt
from operator import add

bufferSize=2**11
sampleRate=48100

p=pyaudio.PyAudio()
chunks=[]
ffts=[]
fig = plt.figure()
ax = fig.add_subplot(111)
li, = ax.plot(np.arange(1000),np.random.randn(1000))
fig.canvas.draw()
plt.ion()
plt.show()

def stream():
	global chunks, inStream, bufferSize
	while True:
		chunks.append(inStream.read(bufferSize))

def record():
	global inStream, bufferSize, p
	inStream=p.open(format=pyaudio.paInt16,channels=1,\
		rate=sampleRate, input=True, frames_per_buffer=bufferSize)
	threading.Thread(target=stream).start()

def smooth(ffty, dgr=2):
	ffts = []
	for i in range(0, len(ffty), dgr):
		ffts.append(sum(ffty[i:i+dgr])/dgr)
	return ffts

def fourier():
	global chunks, bufferSize, w, li, fig
	# Lousy code for fast implementation
	first = 0
	while True:
		if len(chunks)>0:
			data = chunks.pop(0)
			data = scipy.array(struct.unpack("%dB"%(bufferSize*2),data))
			ffty = scipy.fftpack.fft(data)
			fftx = scipy.fftpack.rfftfreq(bufferSize*2, 1.0/sampleRate)
			fftx = fftx[0:len(fftx)/2]
			ffty1 = ffty[:len(ffty)/2]
			ffty2 = ffty[len(ffty)/2:]
			ffty1 = np.abs(ffty1)
			ffty2 = np.abs(ffty2)
			#smooth out the data.
			ffty1 = smooth(ffty1, 4)
			ffty2 = smooth(ffty2, 4)
			ffty = map(add, ffty1, ffty2)
			#Apparently there is noise of 2dB
			ffty = np.log10(ffty) - 2
			fftx = smooth(fftx, 4)
			if first == 0:
				li.set_xdata(fftx)
				li.set_ydata(ffty)
				ax.relim()
				ax.autoscale_view(True, True, True)	
				first = 1

			li.set_ydata(ffty)
			fig.canvas.draw()
		if len(chunks)>20:
			print "This is slow",len(chunks)

def graph():
	global fig
	while(True):
		fig.canvas.draw()
		time.sleep(0.01)

t1 = threading.Thread(target=record).start()
fourier()
