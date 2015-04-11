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
plt.ylim(-5,20)
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

	#take 1. Use global average
def cancelNoiseGlobalAvg(data):
	average = 0
	average = sum(data) / len(data)
	
	#simply subtract the average
	data[:] = [x - average for x in data]

	#take 2. Use local avg.
def cancelNoiseLocalAvg(data):
	average = [0]
	frequencyHops = len(data)/15
	for i in range(len(data)):
		# Every frequencyHops frequency hops we take avg
		if i % frequencyHops == 0:	
			average[len(average)-1] = average[len(average)-1] / frequencyHops
			average.append(0)		
		# Add the them values together	
		average[len(average)-1] = average[len(average)-1] + data[i];

	average[:] = average[1:]
	for i in range(len(data)):
		data[i] = data[i] - average[int(i/frequencyHops)]
	
	#take 3. Use local avg but smoothed
def cancelNoiseSmoothLocalAvg(data):
	margin = 20
	actualElems = 0;
	average = [0]*len(data)
	for i in range(len(data)):
		actualElems = 0
		for j in range(-margin, margin):
			if i+j >= 0 and i+j < len(data):
				average[i] = average[i] + data[i+j]
				actualElems += 1
		if actualElems != 0:
			average[i] = average[i] / actualElems

	peakTshld = 1.1
	for i in range(len(data)):
		if data[i] - average[i]*peakTshld < 0:
			data[i] = 0
		else:
			data[i] = data[i] - average[i]

def fourier():
	global chunks, bufferSize, w, li, fig
	# Lousy code for fast implementation
	first = 0
	while True:
		if len(chunks)>0:
			data = chunks.pop(0)
			data = np.fromstring(data, dtype=np.int16)
			#data = scipy.array(struct.unpack("%dB"%(bufferSize*2),data))

			fft = np.fft.fft(data)
			fftr = 10*np.log10(abs(fft.real))[:len(data)/2]
			ffti = 10*np.log10(abs(fft.imag))[:len(data)/2]
			fftb = 10*np.log10(np.sqrt(fft.imag**2 + fft.real**2))[:len(data)/2]
			freq = np.fft.fftfreq(len(data),1.0/sampleRate)
			freq = freq[0:len(freq)/2]

			#ffty = scipy.fftpack.fft(data)
			#ffty = abs(ffty[0:len(ffty)/2])
			#fftx = scipy.fftpack.rfftfreq(data.size, 1.0/sampleRate)
			##Only positive frequencies
			#fftx = fftx[0:len(fftx)/4]
			##separate positive and negative frequencies
			#ffty1 = ffty[0:len(ffty)/2]
			#ffty2 = ffty[len(ffty)/2:]
			##compute the amplitude of both
			#ffty1 = np.abs(ffty1)
			#ffty2 = np.abs(ffty2)
			##invert the imaginary/complex ones
			#ffty2 = ffty2[::-1]
			##smooth out the data.
			#ffty1 = smooth(ffty1, 4)
			#ffty2 = smooth(ffty2, 4)
			#Add them up
			#ffty = map(add, ffty1, ffty2)	
			#ffty = ffty1 + ffty2
			#Log scale
			#ffty = np.log10(ffty)	
			#cancelNoiseGlobalAvg(fftb)	
			cancelNoiseSmoothLocalAvg(fftb)

			#Smooth the scale as well
			#fftx = smooth(fftx, 4)
			if first == 0:
				li.set_xdata(freq)
				li.set_ydata(fftb)
				ax.relim()
				ax.autoscale_view(True, True, True)	
				first = 1

			li.set_ydata(fftb)
			fig.canvas.draw()
		if len(chunks)>20:
			print "This is slow",len(chunks)

t1 = threading.Thread(target=record).start()
fourier()
