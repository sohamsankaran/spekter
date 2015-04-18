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

#Dropping the buffer size increases the time-resolution. 
#Time resolution = bufferSize / sampleRate (current is about 0.006 sec in theory (!!))
#With that resolution, do we even need synchro?
bufferSize=2**8
sampleRate=44200

#microphone Handle
p=pyaudio.PyAudio()
#Holds chunks of recorded data
chunks=[]
#Holds the frequencies that carry data
freqs = [2000, 5200]
#Holds the indicators for frequencies within the frequencies from fft! 
#see fourier() -- those are the indeces of frequencies
freqIndic = []
#Holds data for stats-graphs -- at the time we don't plot anything
statsY = []
statsGraphY2k = []
statsGraphY5k = []
#See the recordData for details
recordCount = 0
data = []
#Has to be known by the generator. Code for the start/end!
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
		
#Stream the recorded data. Put them in buffer that fourier() then consumes
def stream():
	global chunks, inStream, bufferSize
	while True:
		chunks.append(inStream.read(bufferSize))

#Init. the microphone and start recording
def record():
	global inStream, bufferSize, p
	inStream=p.open(format=pyaudio.paInt16,channels=1,\
		rate=sampleRate, input=True, frames_per_buffer=bufferSize)
	threading.Thread(target=stream).start()

#For noise cancellation I had three attempts, each building on the former. Leave them in
#for project's sake. I will describe those in the report as well.
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
		# Add the values together	
		average[len(average)-1] = average[len(average)-1] + data[i];

	average[:] = average[1:]
	for i in range(len(data)):
		data[i] = data[i] - average[int(i/frequencyHops)]
	
	#take 3. Use local avg but smoothed. This is what we use at the end
def cancelNoiseSmoothLocalAvg(data):
	margin = 10
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
	
	#Annihilate the peaks that are less than peakTshld average. Takes care of most noise
	peakTshld = 1.1
	for i in range(len(data)):
		if data[i] - average[i]*peakTshld < 0:
			data[i] = 0
		else:
			data[i] = data[i] - average[i]

#Record meaningful data that will later be parsed. Begin recording when we have 
#startFreq high enough for 5 consecutive readings. We stop recording when we have
#Another 5 consecutive readings. Then we analyze what we recorded.
def recordData(ffty):
	global recordCount, data

	if recordCount == 5:
		#Because spam is fun!
		print "recording!"	

	#Not recording, waiting for start signal
	if recordCount < 5:
		data.append(ffty)
		#Not to overflow the buffer / trash the memory
		if len(data) > 10:
			data.pop(0)
		
		#We want a continuous start signal to start recording!
		#The 10 is purely empirical. I believe I could even make it 20
		if data[len(data)-1][freqIndic[2]] > 10.0:
			recordCount += 1
		else:
			#If non-continuous, reset counter
			recordCount = 0
	#We are recording, waiting for the stop signal
	elif recordCount < 9:
		data.append(ffty)
		
		#Check for stop signal
		if data[len(data)-1][freqIndic[3]] > 10.0:
			recordCount += 1
		else:
			recordCount = 5
	#Recording stopped, we have the data, now we want to retrieve bits etc.
	elif recordCount == 9:
		print "stopped recording!"
		recordCount = 0
		analyze()
			
#Retrieves bits from the recorded chunks of sound. Then calls convertBinaryToAscii
#To retrieve the actual data
def analyze():
	global data, chunks
	
	print "Started input analysis! Suppressing recorded data"
	#See smoothData for details, but this smooths the signal in time for all freqs
	data = smoothData(data)
	output = ""
	while data:
			
		currentChunk = data.pop(0)
		if sum(currentChunk[freqIndic[0]-1:freqIndic[0]+2]) > 15.0:
			output += "0"
			# So that you do not duplicate the same bit, delete the chunks
			# that are within the same "0" signal! This works because
			# the signal is reasonably smooth (!!!)
			while sum(currentChunk[freqIndic[0]-1:freqIndic[0]+2]) > 3.0:
				currentChunk = data.pop(0)
	
		elif sum(currentChunk[freqIndic[1]-1:freqIndic[1]+2]) > 15.0:
			output += "1"
			# So that you do not duplicate the same bit, delete the chunks
			# that are within the same "1" signal! This works because
			# the signal is reasonably smooth (!!!)
			while sum(currentChunk[freqIndic[1]-1:freqIndic[1]+2]) > 3.0:
				currentChunk = data.pop(0)
	print output
	convertBinaryToAscii(output)

#Smooth data in time for every frequency. Consider current chunk, one in past and one in future
# and take average of these as your value for every frequency. This prevents the signal 
# corresponding to "1" or "0" to have a "gorge" within it that would split it into two
# "1" or "0" signals!
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

#Print out the ascii value of retrieved bits
def convertBinaryToAscii(binary):
	#Only to clean it before going back to normal operation
	global chunks

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
	chunks[:] = [] 

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

#Get the indices for all the frequencies we are using (within freq returned by fft)
def getIndicators(fftFreq):
	global freqIndic, freqs, startFreq, endFreq
	
	resolution = fftFreq[1]-fftFreq[0]	

	#frequencies that carry information
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
			#getStats(fftb)
			#updateStatsGraph()
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
			li.set_ydata(fftb)
			recordData(fftb)
			#fig.canvas.draw()
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
