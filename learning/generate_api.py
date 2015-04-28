#!/usr/bin/env python
import sys
from subprocess import Popen, PIPE
import wave
import math
import struct
import random
import argparse
import binascii
import pdb
import time
from itertools import *
	
class FunctionGenerator:

	#Generate a sine waves
	def sine_wave(self, frequency=440.0, framerate=44100, amplitude=0.5):
		'''
		Generate a sine wave at a given frequency of infinite length.
		'''
		period = int(framerate / frequency)
		if amplitude > 1.0: amplitude = 1.0
		if amplitude < 0.0: amplitude = 0.0
		lookup_table = [float(amplitude) * math.sin(2*math.pi*float(frequency)*(float(i%period)/float(framerate))) for i in xrange(period)]
		return (lookup_table[i%period] for i in count(0))

	#Generate sine wave
	def square_wave(self, frequency=440.0, framerate=44100, amplitude=0.5):
		for s in sine_wave(frequency, framerate, amplitude):
			if s > 0:
				yield amplitude
			elif s < 0:
				yield -amplitude
			else:
				yield 0.0

	#Generate an exponentially damped wave
	def damped_wave(self, frequency=440.0, framerate=44100, amplitude=0.5, length=44100):
		if amplitude > 1.0: amplitude = 1.0
		if amplitude < 0.0: amplitude = 0.0
		return (math.exp(-(float(i%length)/float(framerate))) * s for i, s in enumerate(sine_wave(frequency, framerate, amplitude)))

	#guasian white noise
	def white_noise(amplitude=0.5):
		'''
		Generate random samples.
		'''
		return (float(amplitude) * random.uniform(-1, 1) for i in count(0))

# endFreq -- start and end signal frequency
# freq -- channel frequencels. 2 freqs per channel correspond to 0 and 1
# EOWFreq -- end of word frequency. Sent after every char. For error correction / frame sliding
	# Also doubles as Ack
# sleepTime -- time between every bit sent
# proc -- system process for playing a sound -- linux implementation using "aplay"

class Transmitter:

	funcGen = FunctionGenerator()	
	proc = 0

	def __init__(self, filename = "y", freqs = [2300, 3300, 4000, 4400, 4800, 5400, 9000 ,11300], amplitude = 0.5, startFreq = 2100, endsTime = 0.1, EOWFreq = 1830, framerate = 44100, duration = 0.1, sleepTime = 0.1, bits = 16):
		self.filename = filename
		self.freqs = freqs
		self.rate = framerate
		self.duration = duration
		self.channels = len(freqs)/2
		self.bits = bits
		self.startFreq = startFreq
		self.amplitude = amplitude
		self.endsTime = endsTime
		self.simultaneous = len(freqs)/2
		self.sleepTime = sleepTime
		self.EOWFreq = EOWFreq

	#Used to split memory into chunks to increase performance. 
	def grouper(self, n, iterable, fillvalue=None):
		"grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
		args = [iter(iterable)] * n
		return izip_longest(fillvalue=fillvalue, *args)

	#From the iterative infinite wave get samples
	def compute_samples(self, channels, nsamples=None):
		'''
		create a generator which computes the samples.

		essentially it creates a sequence of the sum of each function in the channel
		at each sample in the file for each channel.
		'''
		b = (imap(sum, izip(*channel)) for channel in channels)
		b = islice(izip(*b), nsamples)
		return b 

	#Write to a .wave file
	def write_wavefile(self, filename, samples, nframes=None, nchannels=2, sampwidth=2, framerate=44100, bufsize=2048, n=0):
		"Write samples to a wavefile."
		if nframes is None:
			nframes = -1
		w = wave.open(filename+str(n), 'w')
		w.setparams((nchannels, sampwidth, framerate, nframes, 'NONE', 'not compressed'))

		max_amplitude = float(int((2 ** (sampwidth * 8)) / 2) - 1)

		# split the samples into chunks (to reduce memory consumption and improve performance)
		for chunk in self.grouper(bufsize, samples):
			frames = ''.join(''.join(struct.pack('h', int(max_amplitude * sample)) for sample in channels) for channels in chunk if channels is not None)
			w.writeframesraw(frames)

		w.close()
		return filename+str(n)

	#Get message from the user and return it in binary
	def getMessageFromInput(self):
		message = raw_input("What message would you like to send?")
		binary = '0' + bin(int(binascii.hexlify(message), 16))[2:]
		return binary

	def sendSingleSignal(self, freq = None, duration = None, filename = None):
		
		if not freq:
			freq = self.freqs[0]
		if not duration:
			duration = self.duration
		if not filename:
			filename = self.filename

	#	freq = freq/2
		#generate signal
		channels = ((self.funcGen.sine_wave(freq, self.rate, self.amplitude),) for i in range(1))
		#length of signal
		samples = self.compute_samples(channels, self.rate * duration)
		#write and play
		self.write_wavefile(filename, samples, self.rate * duration, 1 , self.bits / 8, self.rate)
		if self.proc != 0:
			self.proc.wait()
			self.proc = 0
		#This is just to make sure we don't cut the last signal off. Probably preemptory
		self.proc = Popen(["aplay", filename + "0"])
		time.sleep(duration)

	def sendAllKnownSignals(self, duration = None):
		if duration == None:
			duration = self.duration*4

		#which file to write to / from which file to read
		fqs = self.freqs[:]
		fqs.insert(0, self.startFreq)
		#Notice: false -> 0, 2, 4 etc. true -> 1, 3, 5 etc. (false - 0, true - 1)
		channels = ((self.funcGen.sine_wave(fqs[i], self.rate, self.amplitude),) for i in range(len(fqs)))
		samples = self.compute_samples(channels, self.rate * duration)
		self.write_wavefile("AllKnown", samples, self.rate * duration, len(fqs), self.bits / 8, self.rate)
		
		if self.proc != 0:
			self.proc.wait()
			self.proc = 0
			time.sleep(self.sleepTime)
				
		self.proc = Popen(["aplay", "AllKnown" + "0"])
		time.sleep(duration)

	def sendBits(self, binary, simultaneous = None, sleepTime = None, freqs = None, duration = None, filename = None):
		
		if not simultaneous:
			simultaneous = self.simultaneous
		if not sleepTime:
			sleepTime = self.sleepTime
		if not freqs:
			freqs = self.freqs
		if not duration:
			duration = self.duration
		if not filename:
			filename = self.filename

		n = 0
		print "sending", binary
		while binary:
			binaryList = []
			for i in range(len(self.freqs)/2):
				binaryList.append(binary[:8])
				binary = binary[8:]

				if binary == []:
					break	

			#SFilter any null characters
			binaryList = filter(None, binaryList)

			for i in range(8):
				
				subBinaryList = []
				for j in range(len(binaryList)):
					subBinaryList.append(binaryList[j][0])
					binaryList[j] = binaryList[j][1:]

				#Notice: false -> 0, 2, 4 etc. true -> 1, 3, 5 etc. (false - 0, true - 1)
				channels = ((self.funcGen.sine_wave(freqs[(2*i)+int(subBinaryList[i])], self.rate, self.amplitude),) for i in range(len(subBinaryList)))
				samples = self.compute_samples(channels, self.rate * duration)
				self.write_wavefile(filename, samples, self.rate * duration, len(subBinaryList), self.bits / 8, self.rate, n=n)
				
				if self.proc != 0:
					self.proc.wait()
					self.proc = 0
					time.sleep(self.sleepTime)
				
				self.proc = Popen(["aplay", filename + str(n)])
				n = (n + 1) % 2
		
			#End of words!
			self.sendSingleSignal(freq = self.EOWFreq, duration = 1.2*self.duration)
		
	def sendMessage(self):
		binary = self.getMessageFromInput()
		self.sendSingleSignal(freq = self.startFreq, duration = self.endsTime, filename = "start")
		self.sendBits(binary = binary)
		self.sendSingleSignal(freq = self.startFreq, duration = self.endsTime, filename = "stop")

