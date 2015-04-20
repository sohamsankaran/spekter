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

#For noise cancellation I had three attempts, each building on the former. Leave them in
#for project's sake. I will describe those in the report as well.
class NoiseFilter:
	
		#take 1. Use global avg.
	def cancelNoiseGlobalAvg(self, data):
		average = 0
		average = sum(data) / len(data)
		
		#simply subtract the average
		data[:] = [x - average for x in data]
		
		return data

		#take 2. Use local avg.
	def cancelNoiseLocalAvg(self, data):
		average = [0]
		frequencyHops = len(data)/15
		for i in range(len(data)):
			# Every frequencyHops frequency hops we take avg
			if i % frequencyHops == 0:	
				average[len(average)-1] = average[len(average)-1] / frequencyHops
				average.append(0)		
			# Add the values together	
			average[len(average)-1] = average[len(average)-1] + data[i];

		average[:] = average[1:]
		for i in range(len(data)):
			data[i] = data[i] - average[int(i/frequencyHops)]
		return data		

		#take 3. Use local avg but smoothed. This is what we use at the end
	def cancelNoiseSmoothLocalAvg(self, data):
		#How many elements on each side we will consider for each point
		margin = 10
		#We can't always consider +/- margin (edge cases)! This is how many we 
		#actually consider
		actualElems = 0;
		average = [0]*len(data)
		for i in range(len(data)):
			actualElems = 0
			#Just as before but take average of elems on both sides of the current one
			for j in range(-margin, margin):
				if i+j >= 0 and i+j < len(data):
					average[i] = average[i] + data[i+j]
					actualElems += 1
			if actualElems != 0:
				average[i] = average[i] / actualElems
		
		#Annihilate the peaks that are less than peakTshld*average. Takes care of most noise
		peakTshld = 1.1
		for i in range(len(data)):
			if data[i] - average[i]*peakTshld < 0:
				data[i] = 0
			else:
				data[i] = data[i] - average[i]
		return data

## Description of variables within Receiver:

	# bufferSize -- samples in buffer
	# sampleRate -- rate of sampling in samples per sec
		#Notice: time resolution = bufferSize / sampleRate. Default gives ~0.006 sec
	# freqs -- an array of frequencies that carry information (0,1)
	# startFreq -- frequency of the "Start" signal
	# endFreq -- frequency of the "end" signal
	# chunks -- recorded chunks of data, array
	# p -- microphone handle
	# inStream -- stream from microphone
	# recordedMsg -- recorded non-decrypted message. In the fft chunks format
	# recordThsld -- number of chunks of high start/end freq to star/end recording
	# startEndChunks -- number of consecutive start/end chunks
	# recording -- boolean value - 1 if recording 0 otherwise
	# botBound -- in dB, lower bound to detect signal
	
class Receiver:	
	#For clarity
	bufferSize = 0
	sampleRate = 0
	freqs = []
	freqIndic = []	
	startFreq = 0
	endFreq = 0
	recordCount = 0
	recordedMsg = []
	recordThsld = 5
	botBound = 10.0
	startEndChunks = 0
	p = pyaudio.PyAudio()
	inStream = 0
	chunks = []
	recording = 0
	noise = NoiseFilter()

	def __init__(self, bufferSize = 2**8, sampleRate = 44200, freqs = [2000, 5200], \
			startFreq = 1230, endFreq = 1260):
		self.bufferSize = bufferSize
		self.sampleRate = sampleRate
		self.freqs = freqs
		self.startFreq = startFreq
		self.endFreq = endFreq
		self.inStream = self.p.open(format=pyaudio.paInt16, channels=1,\
			rate=sampleRate, input=True, frames_per_buffer=bufferSize)

	def startMicrophone(self):
		while True:
			self.chunks.append(self.inStream.read(self.bufferSize))

	def start(self):
		threading.Thread(target=self.startMicrophone).start()
		self.fourier()

	def fourier(self):
		# Lousy code for fast implementation
		first = 0
		count = 0
		while True:
			if len(self.chunks)>0:
				data = self.chunks.pop(0)
				data = np.fromstring(data, dtype=np.int16)
				#data = scipy.array(struct.unpack("%dB"%(bufferSize*2),data))

				fft = np.fft.fft(data)
				fftr = 10*np.log10(abs(fft.real))[:len(data)/2]
				ffti = 10*np.log10(abs(fft.imag))[:len(data)/2]
				fftb = 10*np.log10(np.sqrt(fft.imag**2 + fft.real**2))[:len(data)/2]
				fftb = self.noise.cancelNoiseSmoothLocalAvg(fftb)
				#For plotting statistics about significant frequencies
				#getStats(fftb)
				#updateStatsGraph()
				if first == 0:
					#Strong assumption: The frequency-brackets DO NOT CHANGE
					# true for constant chunks
					freq = np.fft.fftfreq(len(data),1.0/self.sampleRate)
					freq = freq[0:len(freq)/2]
					self.getIndicators(freq)
					first = 1
				self.recordMsg(fftb)
			if len(self.chunks)>20:
				print "This is slow",len(self.chunks)

	#Get the indices for all the frequencies we are using (within freq returned by fft)
	def getIndicators(self,fftFreq):
		
		resolution = fftFreq[1]-fftFreq[0]	

		#frequencies that carry information
		for i in range(len(self.freqs)):
			for j in range(len(fftFreq)):
				if np.abs(self.freqs[i] - fftFreq[j]) < resolution:
					self.freqIndic.append(j)
					break
		
		#start freqIndic is one to last
		for j in range(len(fftFreq)):
				if np.abs(self.startFreq - fftFreq[j]) < resolution:
					self.freqIndic.append(j)
					break

		#end freIndic is last
		for j in range(len(fftFreq)):
				if np.abs(self.endFreq - fftFreq[j]) < resolution:
					self.freqIndic.append(j)
					break


	def shouldRecord(self, data):
		#Nor recording, waiting for start signal
		if self.recordCount < self.recordThsld and self.recording == 0:
			if data[self.freqIndic[2]] > self.botBound:
				self.recordCount += 1
				return False
			else:
				self.recordCount = 0
				return False
		#Recording, wait for stop signal
		elif self.recordCount < self.recordThsld and self.recording == 1:
			if data[self.freqIndic[3]] > self.botBound:
				self.recordCount += 1
				return True
			else:
				self.recordCount = 0
				return True
		#Still recording bt you should stop. 
		elif self.recording == 1:
			self.recording = 0
			self.recordCount = 0
			return True
		#Not recording but we should record
		elif self.recording == 0:
			print "asd"
			self.recording = 1
			self.recordCount = 0
			return True

	#Record meaningful data that will later be parsed. Begin recording when we have 
	#startFreq high enough for 5 consecutive readings. We stop recording when we have
	#Another 5 consecutive readings. Then we analyze what we recorded.
	def recordMsg(self, ffty):
		if self.shouldRecord(ffty):
			print "recording!"
			self.recordedMsg.append(ffty)
			#we just stopped
			if self.recording == 0:
				print "Stopped recording! Analyze it!"
				self.analyze()

	#Retrieves bits from the recorded chunks of sound. Then calls convertBinaryToAscii
	#To retrieve the actual data
	def analyze(self):
		
		print "Started input analysis! Suppressing recorded data"
		#See smoothData for details, but this smooths the signal in time for all freqs
		self.recordedMsg = self.smoothData(self.recordedMsg)
		output = ""
		while self.recordedMsg:
				
			currentChunk = self.recordedMsg.pop(0)
			if sum(currentChunk[self.freqIndic[0]-1:self.freqIndic[0]+2]) > self.botBound:
				output += "0"
				# So that you do not duplicate the same bit, delete the chunks
				# that are within the same "0" signal! This works because
				# the signal is reasonably smooth (!!!)
				while sum(currentChunk[self.freqIndic[0]-1:self.freqIndic[0]+2]) > 3.0:
					currentChunk = self.recordedMsg.pop(0)
		
			elif sum(currentChunk[self.freqIndic[1]-1:self.freqIndic[1]+2]) > self.botBound:
				output += "1"
				# So that you do not duplicate the same bit, delete the chunks
				# that are within the same "1" signal! This works because
				# the signal is reasonably smooth (!!!)
				while sum(currentChunk[self.freqIndic[1]-1:self.freqIndic[1]+2]) > 3.0:
					currentChunk = self.recordedMsg.pop(0)
		print output
		self.convertBinaryToAscii(output)

	#Smooth data in time for every frequency. Consider current chunk, one in past and one in future
	# and take average of these as your value for every frequency. This prevents the signal 
	# corresponding to "1" or "0" to have a "gorge" within it that would split it into two
	# "1" or "0" signals!
	def smoothData(self, data):
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

	#Print out the ascii value of retrieved bits
	def convertBinaryToAscii(self, binary):
		print "Converting binary to ASCII" 
		#Split into bytes
		splitList = []
		for i in range(len(binary)/8):
			splitList.append(binary[i*8:i*8+8])
		print splitList
		
		#Convert bytes into ints
		integerVals = []
		for i in range(len(splitList)):
			integerVals.append(int(splitList[i], 2))

		print integerVals
		
		#convert ints into ascii. Notice modulo 128! This is to fail gracefully
		# If you you decode wrong
		msg = ""
		for i in range(len(integerVals)):
			msg += str(unichr(integerVals[i]%128))

		print msg
		#Clean chunks before going back!
		self.chunks[:] = [] 

#Holds data for stats-graphs -- at the time we don't plot anything
statsY = []
statsGraphY2k = []
statsGraphY5k = []
#figure with subplots
fig, (statsPlot2k, statsPlot5k, ax) = plt.subplots(3)
li2k, = statsPlot2k.plot(np.arange(1000), np.random.randn(1000))
li5k, = statsPlot5k.plot(np.arange(1000), np.random.randn(1000))
li, = ax.plot(np.arange(1000),np.random.randn(1000))
fig.canvas.draw()
plt.ylim(-5,20)
plt.ion()
plt.show()
		
#For noise cancellation I had three attempts, each building on the former. Leave them in
#for project's sake. I will describe those in the report as well.
	#take 1. Use global average
#For plotting. Don't worry about this too much
def getStats(fftDataY):
	global statsY
	statsY.append(fftDataY)

	if(len(statsY)>10):
		statsY.pop(0)

#don't worry about this. It's for plotting.
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


