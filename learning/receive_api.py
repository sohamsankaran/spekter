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
		margin = 7
		#We can't always consider +/- margin (edge cases)! This is how many we 
		#actually consider
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
	# startFreq -- frequency of the "Start" and "end"  signal
	# chunks -- recorded chunks of data, array
	# p -- microphone handle
	# inStream -- stream from microphone
	# recordedMsg -- recordedd non-decrypted message. In the fft chunks format
	# recordThsld -- number of chunks of high start/end freq to star/end recording
	# startEndChunks -- number of consecutive start/end chunks
	# recording -- boolean value - 1 if recording 0 otherwise
	# botBound -- in dB, lower bound to detect signal
	# timeOut -- number of chunks processed ithout signal to say its timeout
	
class Receiver:	
	#For clarity
	bufferSize = 0
	sampleRate = 0
	#The frequencies of tsender. as seen by fft
	freqs = []
	#The frequencies of the receiver as seen by fft
	fftFreqs = []
	startFreq = 0
	EOWFreq = 0
	recordCount = 0
	recordedMsg = []
	recordThsld = 5
	botBound = 8.0
	startEndChunks = 0
	p = pyaudio.PyAudio()
	inStream = 0
	chunks = []
	recording = 0
	recordingThread = 0
	#Number of chunks processed without signal to say its timeout
	timeOut = 9
	noise = NoiseFilter()
	hardwareRelatedNoise = 0
	printDebug = 0
	#All the plot variables
	showGraphs = 0
	graphFigure = 0
	#Flag showing if we set the x-axis at all
	xData = 0
	sample0Plot = 0
	s0PlotData = []
	s0PlotLine = 0
	sample1Plot = 0
	s1PlotData = []
	s1PlotLine = 0
	generalPlot = 0
	genPlotData = []
	genPlotLine = 0

	def __init__(self, bufferSize = 2**11, sampleRate = 44200, freqs = [], \
			startFreq=0, timeOut = 100, printDebug = 0, showGraphs = 0):
		self.bufferSize = bufferSize
		self.sampleRate = sampleRate
		self.freqs = freqs
		self.startFreq = startFreq
		self.timeOut = timeOut
		self.printDebug = printDebug
		self.showGraphs = showGraphs
		if showGraphs:
			self.initializeGraphs()
		self.inStream = self.p.open(format=pyaudio.paInt16, channels=1,\
			rate=sampleRate, input=True, frames_per_buffer=bufferSize)
		self.startRecording()

	#Buffer the stream
	def startMicrophone(self):
		while True:
			self.chunks.append(self.inStream.read(self.bufferSize))

	#Start recording from the mic
	def startRecording(self):
		if self.printDebug == 1:
			print "Recording started!"
		self.recordingThread = threading.Thread(target=self.startMicrophone)
		self.recordingThread.start()

	def waitForSignal(self, frequency, errorAllowable = 0, purge = 1, timeOut = None):

		#purge buffer
		if purge == 1:
			self.chunks = []
		
		#If we have no clue about frequencies from fft, get them
		if self.fftFreqs == []:
			while len(self.chunks) == 0:
				continue

			#Get indicator...
			data = self.fourier(wantFreqs = 1)
			self.fftFreqs = data[1]
		
		indicator = []

		freqResolution = self.fftFreqs[1]-self.fftFreqs[0]
		for i in range(len(self.fftFreqs)):
			if np.abs(frequency - self.fftFreqs[i]) < freqResolution:
				indicator.append(i)
	
		indicator = math.trunc(sum(indicator)/len(indicator))
	
		if self.printDebug == 1:
			print "The indicator is", indicator, "the frequency there is", self.fftFreqs[indicator], "The real frequency is ", frequency
	
		#Wait for the signal...
		if timeOut == None:
			timeOut = self.timeOut

		tries = 0
		consequentSignals = 0
		errorInRes = math.trunc(errorAllowable / freqResolution)

		if purge == 0:
			cpChunks = []

		while tries < timeOut:

			if purge == 0 and len(self.chunks) > 0:
				cpChunks.append(self.chunks[0])
			#If we have chunks to analyze
			if len(self.chunks) > 0:
				fftb = self.fourier()
				#Increment number of tries
				tries += 1		
				#Check if we got meaningful signal
				#Get the top and bot boundaries of frequency that you use to look for signal
				bot = 0
				if indicator - errorInRes < 0:
					bot = 0
				else:
					bot = indicator - errorInRes

				top = 0
				if indicator + errorInRes > len(fftb):
					top = len(fftb)
				else:
					top = indicator + errorInRes

				maxStr = max(fftb[bot : top + 1])
				
				if self.printDebug == 1:
					print "The strength of the signal is ", maxStr

				if (maxStr > self.botBound):
					consequentSignals += 1
				else:
					consequentSignals = 0
		
			#If we decided that we did get the right stuff, return 1!
			if consequentSignals > self.recordThsld:
				if self.printDebug == 1:
					print "Got the frequency signal!", frequency

				if purge == 0:
					self.chunks = cpChunks + self.chunks
				return 1

		#If we time out, just return false
		if self.printDebug == 1:
			print "Timed out waiting for signal!", frequency

			if purge == 0:
				self.chunks = cpChunks + self.chunks
		return 0
	
	def snoop(self, timeOut = None, signal = None, purge = 1):
		
		if purge == 1:
			self.chunks = []

		if timeOut == None:
			timeOut = self.timeOut

		if self.fftFreqs == []:
			#Wait for at least one chunk recorded
			while len(self.chunks) == 0:
				continue

			#Get indicator...
			data = self.fourier(wantFreqs = 1)
			self.fftFreqs = data[1]
		
		freqResolution = self.fftFreqs[1] - self.fftFreqs[0]
			
		#Wait for the signal...
		tries = 0
		consequentSignals = [0] * len(self.fftFreqs)
		while tries < timeOut:
			#If we have chunks to analyze
			if len(self.chunks) > 0:
				fftb = self.fourier()
				#Increment number of tries
				tries += 1		
				#Check if we got meaningful signal
				for j in range(len(fftb)):
					if fftb[j] > self.botBound:
						if signal == None:
							consequentSignals[j] += 1
						if self.printDebug == 1:
							print "Got a strong ping: ", self.fftFreqs[j], "Of strength ", fftb[j] 
						if consequentSignals[j] > self.recordThsld:
							if self.printDebug == 1:
								print "Got a strong signal!", self.fftFreqs[j]
							return (1, self.fftFreqs[j])
					else:
						consequentSignals[j] = 0
			if len(self.chunks) > 20 and self.printDebug == 1:
				print "Snooping is slow! I have ", len(self.chunks), " chunks already..."

		#If we time out, just return false
		if self.printDebug == 1:
			print "Timed out waiting for a strong signal!"
		return (0, 0)

	#Need estimates of synchro. Sender can always provide
	def getFreqs(self, synchroFreq, startFreq, freqs, estimatedError = 200):
		freqResolution = self.fftFreqs[1] - self.fftFreqs[0]
		estimatedErrorFrame = math.trunc(estimatedError/freqResolution)
		indicatorSync = []
		indicatorStart = []
		indicatorFreqs = []
		for i in range(len(freqs)):
			indicatorFreqs.append([])

		#Get the indicator for sunchro
		for i in range(len(self.fftFreqs)):
			if np.abs(synchroFreq - self.fftFreqs[i]) < freqResolution:
				indicatorSync.append(i)
				continue
			if np.abs(startFreq - self.fftFreqs[i]) < freqResolution:
				indicatorStart.append(i)
				continue
			for j in range(len(indicatorFreqs)):
				if np.abs(freqs[j] - self.fftFreqs[i]) < freqResolution:
					indicatorFreqs[j].append(i)
					break

		indicatorSync = math.trunc(sum(indicatorSync)/len(indicatorSync))
		indicatorStart = math.trunc(sum(indicatorStart)/len(indicatorStart))
	
		for i in range(len(indicatorFreqs)):
			indicatorFreqs[i] = math.trunc(sum(indicatorFreqs[i])/len(indicatorFreqs[i]))
		#get the lower bound for synchro freq
		botSync = 0
		botStart = 0
		botFreqs = [0]*len(indicatorFreqs)
		if indicatorSync > estimatedErrorFrame:
			botSync = indicatorSync - estimatedErrorFrame
		if indicatorStart > estimatedErrorFrame:
			botStart = indicatorStart - estimatedErrorFrame
		for i in range(len(botFreqs)):
			if indicatorFreqs[i] > estimatedErrorFrame:
				botFreqs[i] = indicatorFreqs[i] - estimatedErrorFrame

		#Get the top bound for synchro freq
		topSync = len(self.fftFreqs)
		topStart = len(self.fftFreqs)
		topFreqs = [0]*len(indicatorFreqs)		

		if indicatorSync + estimatedErrorFrame < len(self.fftFreqs):
			topSync = indicatorSync + estimatedErrorFrame
		if indicatorStart + estimatedErrorFrame < len(self.fftFreqs):
			topStart = indicatorStart + estimatedErrorFrame
		for i in range(len(topFreqs)):
			if indicatorFreqs[i] + estimatedErrorFrame < len(self.fftFreqs):
				topFreqs[i] = indicatorFreqs[i] + estimatedErrorFrame

		#"Histogram" of freqs > botBound around the synchro region
		synchroFreqs = [0]*(topSync - botSync + 1)
		startFreqs = [0]*(topStart - botStart + 1)
		freqFreqs = []
		for i in range(len(topFreqs) - 1, -1, -1):
			freqFreqs.append([0]*(topFreqs[i] - botFreqs[i] + 1))

		#Go through all frames and fill in histogram
		for j in range(len(self.recordedMsg)):
			curr = self.recordedMsg[j]
			for i in range(botSync, topSync):
				if curr[i] > self.botBound-4:
					synchroFreqs[i-botSync] += 1

			for i in range(botStart, topStart):
				if curr[i] > self.botBound-4:
					startFreqs[i - botStart] += 1

			for k in range(len(freqFreqs)):
				for i in range(botFreqs[k], topFreqs[k]):
					if curr[i] > self.botBound-4:
						freqFreqs[k][i - botFreqs[k]] += 1
		#Get the synchro signal from the histogram. Assume it is the strongest
		topValue = 0
		topIndic = 0
		for j in range(len(synchroFreqs)):
			if synchroFreqs[j] >= topValue:
				topValue = synchroFreqs[j]
				topIndic = botSync + j
			
		self.EOWFreq = topIndic
		
		if self.printDebug == 1:
			print "Found synchro frequency. It is about: ", self.fftFreqs[topIndic], "out of", synchroFreqs

		topValue = 0
		topIndic = 0
		for j in range(len(startFreqs)):
			if startFreqs[j] >= topValue:
				topValue = startFreqs[j]
				topIndic = botStart + j

		self.startFreq = topIndic

		if self.printDebug == 1:
			print "Found start frequency. It is about: ", self.fftFreqs[topIndic], "out of", startFreqs

		for i in range(len(freqFreqs)):
			topValue = 0
			topIndic = 0
			for j in range(len(freqFreqs[i])):
				if freqFreqs[i][j] >= topValue:
					topValue = freqFreqs[i][j]
					topIndic = botFreqs[i] + j

			self.freqs.append(topIndic)

			if self.printDebug == 1:
				print "Found important frequency. It is about ", self.fftFreqs[topIndic], "out of", freqFreqs[i]
		

##Delete multiples of EOWFreq from input. SIC. Note: can delete up to 2 frames per iteration
#		if SIC == 1:
#			for j in range(len(self.recordedMsg)):
#				multi = self.EOWFreq
#				while multi < 20000:
#					#Get the indicator for multi, delete that value
#					for i in range(len(self.fftFreqs)):
#						if np.abs(multi - self.fftFreqs[i]) < freqResolution:
#							self.recordedMsg[j][i] = 0
#					multi += multi
#		pdb.set_trace()
#			#Smooth the current input in time. All of it.
#		self.recordedMsg = self.smoothData(self.recordedMsg, smoothFactor = 2*len(self.recordedMsg))
#		pdb.set_trace()
#		if self.showGraphs == 1:
#			self.setGraphData(self.generalPlot, self.genPlotLine, self.recordedMsg[math.trunc(len(self.recordedMsg)/2)], relim = 0)
#			plt.savefig('LearningSignalsAfterSmooth.jpg')
	
		#Go through the thing. Once you find a frequency band, add it to freqs and 
		#Delete the multiples. Look only at middle frame. All should be the same after smooth
#		mid = math.trunc(len(self.recordedMsg)/2)
#		pdb.set_trace()
#		for i in range(len(self.fftFreqs)):
#			if self.recordedMsg[mid][i] > self.botBound:
#				#Remember the resolution is not v. high. Check the next frame too!
#				if self.recordedMsg[mid][i+1] > self.recordedMsg[mid][i]:
#					#If the next one is actually better, get the better one and ignore this
#					continue

				#The first found frequency is start. All the ones later are channels.
#				if self.startFreq == 0:
#					if self.printDebug == 1:
#						print "Found start frequency! Its frequency is, ", self.fftFreqs[i]
#
#					self.startFreq = self.fftFreqs[i]	
#				else:
#					if self.printDebug == 1:
#						print "Found important frequency! Its frequency is, ", self.fftFreqs[i]
#			
#					self.freqs.append(self.fftFreqs[i])

				#Now delete all the multiples
#				if SIC == 1:
#					multi = self.fftFreqs[i]
#					while multi < 20000:
#						#Get the inficator for multi, delete that value
#						for i in range(len(self.fftFreqs)):
#							if np.abs(multi - self.fftFreqs[i]) < freqResolution:
#								self.recordedMsg[mid][i] = 0
#					multi += multi

		self.recordedMsg = []
		return len(self.freqs) + 2

	def purgeFreqs(self):
		self.EOWFreq = 0
		self.startFreq = 0
		self.freqs = []
	
	def waitForMessage(self):
		# Lousy code for fast implementation
		count = 0
		while True:
			if len(self.chunks)>0:
				fftb = self.fourier()
				self.recordMsg(fftb)
			if len(self.chunks)>20:
				print "This is slow",len(self.chunks)

	#Caller has to make sure that there exists chunks to be popped!	
	def fourier(self, wantFreqs = 0):
			data = self.chunks.pop(0)
			data = np.fromstring(data, dtype=np.int16)
			#data = scipy.array(struct.unpack("%dB"%(bufferSize*2),data))
			fft = np.fft.fft(data)
			fftr = 10*np.log10(abs(fft.real))[:len(data)/2]
			ffti = 10*np.log10(abs(fft.imag))[:len(data)/2]
			fftb = 10*np.log10(np.sqrt(fft.imag**2 + fft.real**2))[:len(data)/2]
			fftb = self.noise.cancelNoiseSmoothLocalAvg(fftb)

			if self.fftFreqs != []:
				if self.hardwareRelatedNoise == 0:
					i = 0
					while self.fftFreqs[i] < 1600:
						i += 1
					self.hardwareRelatedNoise = i	
					fftb = fftb[i:]
					self.fftFreqs = self.fftFreqs[i:]	
				else:
					fftb = fftb[self.hardwareRelatedNoise:]
						

			if wantFreqs != 0:
				#Strong assumption: The frequency-brackets DO NOT CHANGE
				# true for constant chunks
				freq = np.fft.fftfreq(len(data),1.0/self.sampleRate)
				freq = freq[0:len(freq)/2]
				if self.showGraphs == 1 and self.xData == 0:
					self.s0PlotLine.set_xdata(freq) 
					self.s1PlotLine.set_xdata(len(freq))
					self.genPlotLine.set_xdata(len(freq))
					self.xData = 1
				return (fftb, freq)
			return fftb

	def shouldRecord(self, data):
		#Nor recording, waiting for start signal
		if self.recordCount < self.recordThsld and self.recording == 0:
			if data[self.startFreq] > self.botBound:
				self.recordCount += 1
				return False
			else:
				self.recordCount = 0
				return False
		#Recording, wait for stop signal
		elif self.recordCount < self.recordThsld and self.recording == 1:
			if data[self.startFreq] > self.botBound:
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
			self.recording = 1
			self.recordCount = 0
			return True

	#Record meaningful data that will later be parsed. Begin recording when we have 
	#startFreq high enough for 5 consecutive readings. We stop recording when we have
	#Another 5 consecutive readings. Then we analyze what we recorded.
	def recordMsg(self, ffty):
		if self.shouldRecord(ffty):
			if self.printDebug == 1:
				print "recording!"
			recordInput(ffty)
			#we just stopped
			if self.recording == 0:
				if self.printDebug == 1:
					print "Stopped recording! Analyze it!"
				self.analyze()

	#User is responsible for purging the recording buffer
	def recordInput(self):
		self.chunks = []

	def stopRecordingInput(self):
		self.recordedMsg = []
		for j in range(len(self.chunks)):
			self.recordedMsg.append(self.fourier())

	#Retrieves bits from the recorded chunks of sound. Then calls convertBinaryToAscii
	#To retrieve the actual data
	def analyze(self, possibleError = 4):
		if self.printDebug == 1:
			print "We have ", len(self.recordedMsg), " chunks to analyze!"
		for j in range(len(self.recordedMsg)):
			for i in range(len(self.recordedMsg[j])):
				if self.recordedMsg[j][i] > self.botBound:
					print self.recordedMsg[j][i], self.fftFreqs[i]

		if self.printDebug == 1:	
			print "Started input analysis! Suppressing recorded data"
		#Show a plot of smooth vs. non smooth data. For first carrier, first 200 pts
		if self.showGraphs == 1:
			notSmooth = []
			for i in range(len(self.recordedMsg)):
				if i > 200:
					break
				notSmooth.append[self.recordedMsg[i][self.freqs[0]]]

			self.setGraphData(self.sample0Plot, self.s0PlotLine, notSmooth, \
				range(len(notSmooth)))
		
		#See smoothData for details, but this smooths the signal in time for all freqs
		self.recordedMsg = self.smoothData(self.recordedMsg)

		if self.showGraphs == 1:
			smooth = []
			
			for i in range(len(self.recordedMsg)):
				if i > 200:
					break
				smooth.append[self.recordedMsg[i][self.freqs[0]]]

			self.setGraphData(self.sample0Plot, self.s0PlotLine, smooth, \
				range(len(smooth)))

			plt.savefig("SmoothAndNotSmoothAnalysis.jpg")	
		output = []
		for i in range(len(self.freqs)/2):
			output.append([""])
		#Note that this also purges the record buffer
		while self.recordedMsg:
				
			currentChunk = self.recordedMsg.pop(0)
			
			#End of word!
			if currentChunk[self.EOWFreq] > self.botBound:
				for i in range(1, len(self.freqs)/2+1):
					if len(output[-i]) != 8:
						print "Corrupted bit!"
					#If too few bits, fill them in
					while len(output[-i][0]) < 8:
						output[-i][0] = output[-i][0] + "0"

					#If too many bits, cut them out
					while len(output[-i][0]) > 8:
						output[-i][0] = output[-i][0][:-1]

				for i in range(len(self.freqs)/2):
					output.append([""])
			
				while currentChunk[self.EOWFreq] > self.botBound:
					currentChunk = self.recordedMsg.pop(0)
				
				if self.printDebug == 1:
					print "Got end of word!"	
	
				continue
				
			#Getting rid of multiplicates. Remember one freq. All synchronized in that
			#sense
			
			else: 
				msg = 0
				for i in range(len(self.freqs)):
					if sum(currentChunk[self.freqs[i] - possibleError: self.freqs[i] + possibleError]) > self.botBound:
						if i%2 == 0:
							print i, i/2-len(self.freqs)/2
							output[i/2-len(self.freqs)/2][0] = output[i/2 - len(self.freqs)/2][0] + "0"
						else:
							print i, (i-1)/2 - len(self.freqs)/2
							output[(i-1)/2 - len(self.freqs)/2][0] = output[(i-1)/2-len(self.freqs)/2][0] + "1"
						msg = i
					
				# So that you do not duplicate the same bit, delete the chunks
				# that are within the same "symbol" signal! This works because
				# the signal is reasonably smooth (!!!)

				while sum(currentChunk[self.freqs[msg] - possibleError : self.freqs[msg] + possibleError]) > self.botBound:
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


#	#smoothFactor tells us how to average out / how much to average out -- how many time frames
#	# to take
#	def smoothData(self, data, smoothFactor = 3):
#	
#		smoothedGlobal = []
#		for i in range(len(data)):
#			smoothedLocal = []
#			
#			framesToSmooth = math.trunc(smoothFactor / 2)
#			
#			botBoundForSmoothing = 0
#			if i > framesToSmooth:
#				botBoundForSmoothing = i - framesToSmooth
#
#			topBoundForSmoothing = len(data)
#			if len(data) > i + framesToSmooth:
#				topBoundForSmoothing = i + framesToSmooth
#			
#
#			#This could fail if data frames are different length. Good they are not.
#			for j in range(len(data[i])):
#				tmp = []
#
#				for k in range(botBoundForSmoothing, topBoundForSmoothing):
#					tmp.append(data[k][j])
#				
#				smoothedLocal.append(sum(tmp)/(topBoundForSmoothing - botBoundForSmoothing + 1))
#
#			smoothedGlobal.append(smoothedLocal)
#
#		for i in range(len(smoothedGlobal[0])):
#			if smoothedGlobal[0][i] > 1:
#				print smoothedGlobal[0][i], self.fftFreqs[i]
#	
#		return smoothedGlobal

	#Print out the ascii value of retrieved bits
	def convertBinaryToAscii(self, binary):
		if self.printDebug == 1:
			print "Converting binary to ASCII" 
		
		#Filter out empty strings... basically 4 last elems
		binary = binary[:len(binary)-4]		

		#Convert bytes into ints
		integerVals = []
		for i in range(len(binary)):
			integerVals.append(int(binary[i][0], 2))

		if self.printDebug == 1:
			print integerVals
		
		#convert ints into ascii. Notice modulo 128! This is to fail gracefully
		# If you you decode wrong
		msg = ""
		for i in range(len(integerVals)):
			msg += str(unichr(integerVals[i]%128))
		
		if self.printDebug == 1:
			print msg
		#Clean chunks before going back!
		self.chunks[:] = []

	def initializeGraphs(self):
		self.graphFigure, (self.sample0Plot, self.sample1Plot, self.generalPlot) \
			= plt.subplots(3)
		self.s0PlotLine, = self.sample0Plot.plot(np.arange(1000), np.random.randn(1000))
		self.s1PlotLine, = self.sample1Plot.plot(np.arange(1000), np.random.randn(1000))
		self.genPlotLine, = self.generalPlot.plot(np.arange(1000), np.random.randn(1000))
		self.graphFigure.canvas.draw()
		plt.ylim(-5,20)
		plt.ion()
		plt.show()

	def setGraphData(self, graph, graphLine, yData = None, xData = None, relim = 1):
		if yData != None:
			graphLine.set_ydata(yData)
	
		if xData != None:
			graphLine.set_xdata(xData)

		if relim == 1:
			graph.relim()
			graph.autoscale_view(True, True, True)
