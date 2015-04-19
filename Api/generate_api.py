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

class Transmitter:

	funcGen = FunctionGenerator()	
	proc = 0

	def __init__(self, filename, freqs, amplitude, startFreq, endFreq, endsTime, framerate, time, channels, bits, simultaneous):
		self.filename = filename
		self.freqs = freqs
		self.rate = framerate
		self.time = time
		self.channels = channels
		self.bits = bits
		self.startFreq = startFreq
		self.endFreq = endFreq
		self.amplitude = amplitude
		self.endsTime = endsTime
		self.simultaneous = simultaneous

	#Used to split memory into chunks to increase performance. 
	def grouper(n, iterable, fillvalue=None):
		"grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
		args = [iter(iterable)] * n
		return izip_longest(fillvalue=fillvalue, *args)

	#From the iterative infinite wave get samples
	def compute_samples(channels, nsamples=None):
		'''
		create a generator which computes the samples.

		essentially it creates a sequence of the sum of each function in the channel
		at each sample in the file for each channel.
		'''
		b = islice(izip(*(imap(sum, izip(*channel)) for channel in channels)), nsamples)
		return b 

	#Write to a .wave file
	def write_wavefile(filename, samples, nframes=None, nchannels=2, sampwidth=2, framerate=44100, bufsize=2048, n=0):
		"Write samples to a wavefile."
		if nframes is None:
			nframes = -1
		w = wave.open(filename+str(n), 'w')
		w.setparams((nchannels, sampwidth, framerate, nframes, 'NONE', 'not compressed'))

		max_amplitude = float(int((2 ** (sampwidth * 8)) / 2) - 1)

		# split the samples into chunks (to reduce memory consumption and improve performance)
		for chunk in grouper(bufsize, samples):
			frames = ''.join(''.join(struct.pack('h', int(max_amplitude * sample)) for sample in channels) for channels in chunk if channels is not None)

			w.writeframesraw(frames)

		w.close()
		
		return filename+str(n)

	#Get message from the user and return it in binary
	def getMessageFromInput():
		message = raw_input("What message would you like to send?")
		binary = '0' + bin(int(binascii.hexlify(word), 16))[2:]
		return binary

	def sendSingleSignal(freq = self.freqs[0], time = self.time, filename = self.filename):
		freq = freq/2
		#generate signal
		channels = ((sine_wave(freq, rate, amplitude),) for i in range(1))
		#length of signal
		samples = compute_samples(channels, rate * time)
		#write and play
		write_wavefile(filename, samples, rate * time, channels, bits / 8, rate)
		self.proc = Popen(["aplay", filename + "0"])

	def sendBits(binary, simultaneous = self.simultaneous, sleepTime = self.sleepTime, freqs = self.freqs, time = self.time, filename = self.filename):
		#which file to write to / from which file to read
		n = 0
		print "sending", binary
		while binary:
			if (len(binary) < 4*simultaneous):
				p = len(binary)/4+1
			else:
				p = simultaneous
			
			binaryList = []
			for i in range(p):
				binaryList.insert(len(binaryList), binary[:4])
				binary = binary[4:]

			#SFilter any null characters
			binaryList = filter(None, binaryList)

			while binaryList:
				#we keep the first bits within each word
				subBinaryList = []
				p = len(binaryList)			
				while p:
					#get the next bit
					subBinaryList.insert(0,binaryList[p-1][0])
					#delete it
					binaryList[p-1] = binaryList[p-1][1:]
					#delete empty elements in the list
					if not binaryList[p-1]:
						binaryList.pop(p-1)
					p -= 1
				#Notice: false -> 0, 2, 4 etc. true -> 1, 3, 5 etc. (false - 0, true - 1)
				channels = ((sine_wave(freqs[(2*i)+int(subBinaryList[i])], rate, amplitude),) for i in range(len(subBinaryList)))
				samples = compute_samples(channels, rate * time)
				write_wavefile(filename, samples, rate * time, channels, bits / 8, rate, n=n)
				
				if 'proc' in locals():
					proc.wait()
					time.sleep(sleepTime)
				
				proc = Popen(["aplay", filename + str(n)])
				n = (n + 1) % 2
		
	def sendMessage(self):
		binary = getMessageFromInput()
		sendSingleSignal(freq = startFreq, time = endsTime, filename = "start")
		sendBits(binary = binary)
		sendSingleSignal(freq = endFreq, time = endsTime, filename = "stop")
	
class FunctionGenerator:

	#Generate a sine waves
	def sine_wave(frequency=440.0, framerate=44100, amplitude=0.5):
		'''
		Generate a sine wave at a given frequency of infinite length.
		'''
		period = int(framerate / frequency)
		if amplitude > 1.0: amplitude = 1.0
		if amplitude < 0.0: amplitude = 0.0
		lookup_table = [float(amplitude) * math.sin(2*math.pi*float(frequency)*(float(i%period)/float(framerate))) for i in xrange(period)]
		return (lookup_table[i%period] for i in count(0))

	#Generate sine wave
	def square_wave(frequency=440.0, framerate=44100, amplitude=0.5):
		for s in sine_wave(frequency, framerate, amplitude):
			if s > 0:
				yield amplitude
			elif s < 0:
				yield -amplitude
			else:
				yield 0.0

	#Generate an exponentially damped wave
	def damped_wave(frequency=440.0, framerate=44100, amplitude=0.5, length=44100):
		if amplitude > 1.0: amplitude = 1.0
		if amplitude < 0.0: amplitude = 0.0
		return (math.exp(-(float(i%length)/float(framerate))) * s for i, s in enumerate(sine_wave(frequency, framerate, amplitude)))

	#guasian white noise
	def white_noise(amplitude=0.5):
		'''
		Generate random samples.
		'''
		return (float(amplitude) * random.uniform(-1, 1) for i in count(0))
