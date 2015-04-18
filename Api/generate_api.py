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

#Used to split memory into chunks to increase performance. 
def grouper(n, iterable, fillvalue=None):
	"grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
	args = [iter(iterable)] * n
	return izip_longest(fillvalue=fillvalue, *args)

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
