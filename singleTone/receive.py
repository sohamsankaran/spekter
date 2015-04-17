import pyaudio
import scipy
import struct
import argparse
import scipy.fftpack
import threading
import multiprocessing
import time, datetime
import math
import pdb
import numpy as np
import matplotlib.pyplot as plt
from operator import add

bufferSize=2**11
sampleRate=44100

#microphone Handle
p=pyaudio.PyAudio()
#Holds chunks of recorded data
chunks=[]
#Holds the frequencies that carry data
freqs = [2000, 5200]
#Holds the indicators for frequencies within the frequencies from fft! 
#see fourier()
freqIndic = []
#Holds data for stats-graphs
statsY = []
statsGraphY2k = []
statsGraphY5k = []
#See the recordData for details
recordCount = 0
data = []
#Has to be known by the generator
startFreq = 1260
endFreq = 1260
#figure with subplots
fig, (statsPlot2k, statsPlot5k, ax) = plt.subplots(3)
li2k, = statsPlot2k.plot(np.arange(1000), np.random.randn(1000))
li5k, = statsPlot5k.plot(np.arange(1000), np.random.randn(1000))
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
	margin = 10
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

def recordData(ffty):
	global recordCount, data

	if recordCount == 5:
		print "recording!"	

	if recordCount < 5:
		data.append(ffty)
		if len(data) > 10:
			data.pop(0)
		
		#We want a continuous start signal to start recording!
		if data[len(data)-1][freqIndic[2]] > 8.0:
			recordCount += 1
		else:
			recordCount = 0
	elif recordCount < 9:
		data.append(ffty)
		
		#Check for stop signal
		if data[len(data)-1][freqIndic[3]] > 8.0:
			recordCount += 1
		else:
			recordCount = 5
	elif recordCount == 9:
		print "stopped recording!"
		recordCount = 0
		analyze()
			
def analyze():
	global data, chunks
	
	print "Started input analysis! Suppressing recorded data"
	data = smoothData(data)
	output = ""
	while data:
			
		currentChunk = data.pop(0)
		if sum(currentChunk[freqIndic[0]-1:freqIndic[0]+2]) > 15.0:
			output += "0"
			print "0"
			# So that you do not duplicate the same bit
			while sum(currentChunk[freqIndic[0]-1:freqIndic[0]+2]) > 3.0:
				print sum(currentChunk[freqIndic[0]-1:freqIndic[0]+2])
				currentChunk = data.pop(0)
	
		elif sum(currentChunk[freqIndic[1]-1:freqIndic[1]+2]) > 15.0:
			output += "1"
			print "1"
			# So that you do not duplicate the same bit
			while sum(currentChunk[freqIndic[1]-1:freqIndic[1]+2]) > 3.0:
				print sum(currentChunk[freqIndic[0]-1:freqIndic[0]+2]) 
				currentChunk = data.pop(0)
	print output
	convertBinaryToAscii(output)

#Smooth data in time for every frequency
def smoothData(data):
	smoothedGlobal = []
	for i in range(1,len(data)-1):
		smoothedLocal = []
		for j in range(len(data[i])):
			past = data[i-1][j]
			now = data[i][j]
			future = data[i+1][j]
			smoothedLocal.append((past + now + future)/3)
		smoothedGlobal.append(smoothedLocal)
	return smoothedGlobal

def convertBinaryToAscii(binary):
	#Only to clean it before going back to normal operation
	global chunks

	print "Converting binary to ASCII" 

	splitList = []
	for i in range(len(binary)/8):
		splitList.append(binary[i*8:i*8+8])
	print splitList	
	integerVals = []
	for i in range(len(splitList)):
		integerVals.append(int(splitList[i], 2))

	print integerVals
	msg = ""
	for i in range(len(integerVals)):
		msg += str(unichr(integerVals[i]))

	print msg
	#Clean chunks before going back!
	chunks[:] = [] 

def getStats(fftDataY):
	global statsY
	statsY.append(fftDataY)

	if(len(statsY)>10):
		statsY.pop(0)

def updateStatsGraph():
	global statsPlot2k, statsPlot5k, statsY, li2, statsGraphY2k, statsGraphY5k, freqIndic
	j = 0
	#i = raw_input('What element of statsY array?')
	p = 0	
	i = freqIndic[0]
	statsGraphY2k.append(sum(statsY[len(statsY)-1][i-1:i+2]))
	i = freqIndic[1]
	statsGraphY5k.append(sum(statsY[len(statsY)-1][i-1:i+2]))
	#To avoid slow down	
	if len(statsGraphY2k) > 100:
		statsGraphY2k.pop(0)
		statsGraphY5k.pop(0)

	Xdata = range(len(statsGraphY2k))
	li2k.set_ydata(statsGraphY2k)
	li2k.set_xdata(Xdata)
	li5k.set_ydata(statsGraphY5k)
	li5k.set_xdata(Xdata)
	statsPlot2k.relim()
	statsPlot2k.autoscale_view(True, True, True)
	statsPlot5k.relim()
	statsPlot5k.autoscale_view(True, True, True)

def getIndicators(fftFreq):
	global freqIndic, freqs, startFreq, endFreq
	
	resolution = fftFreq[1]-fftFreq[0]	

	for i in range(len(freqs)):
		for j in range(len(fftFreq)):
			if np.abs(freqs[i] - fftFreq[j]) < resolution:
				freqIndic.append(j)
				break
	
	#start freqIndic is one to last
	for j in range(len(fftFreq)):
			if np.abs(startFreq - fftFreq[j]) < resolution:
				freqIndic.append(j)
				break

	#end freIndic is last
	for j in range(len(fftFreq)):
			if np.abs(endFreq - fftFreq[j]) < resolution:
				freqIndic.append(j)
				break

def fourier():
	global chunks, bufferSize, w, li, fig, statsPlot
	# Lousy code for fast implementation
	first = 0
	count = 0
	while True:
		if len(chunks)>0:
			data = chunks.pop(0)
			data = np.fromstring(data, dtype=np.int16)
			#data = scipy.array(struct.unpack("%dB"%(bufferSize*2),data))

			fft = np.fft.fft(data)
			fftr = 10*np.log10(abs(fft.real))[:len(data)/2]
			ffti = 10*np.log10(abs(fft.imag))[:len(data)/2]
			fftb = 10*np.log10(np.sqrt(fft.imag**2 + fft.real**2))[:len(data)/2]
			cancelNoiseSmoothLocalAvg(fftb)
			#For plotting statistics about significant frequencies
			getStats(fftb)
			if first == 0:
				#Strong assumption: The frequency-brackets DO NOT CHANGE
				# true for constant chunks
				freq = np.fft.fftfreq(len(data),1.0/sampleRate)
				freq = freq[0:len(freq)/2]
				getIndicators(freq)
				for i in range(len(freqIndic)):	
					print freq[freqIndic[i]]
	
				li.set_xdata(freq)
				li.set_ydata(fftb)
				ax.relim()
				ax.autoscale_view(True, True, True)	
				first = 1
				updateStatsGraph()
			li.set_ydata(fftb)
			updateStatsGraph()
			recordData(fftb)
			fig.canvas.draw()
		if len(chunks)>20:
			print "This is slow",len(chunks)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--frequency', help="Seed for frequencies of the wave in Hz", default = 1000, type=float)
	parser.add_argument('-s', '--step', help="Step for generating next frequencies in Hz", default = 97.0, type=float)

	args = parser.parse_args()
	t1 = threading.Thread(target=record).start()
	fourier()

if __name__ == "__main__":
	main()
