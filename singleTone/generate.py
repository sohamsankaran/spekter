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

def grouper(n, iterable, fillvalue=None):
	"grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
	args = [iter(iterable)] * n
	return izip_longest(fillvalue=fillvalue, *args)

def sine_wave(frequency=440.0, framerate=44100, amplitude=0.5):
	'''
	Generate a sine wave at a given frequency of infinite length.
	'''
	period = int(framerate / frequency)
	if amplitude > 1.0: amplitude = 1.0
	if amplitude < 0.0: amplitude = 0.0
	lookup_table = [float(amplitude) * math.sin(2*math.pi*float(frequency)*(float(i%period)/float(framerate))) for i in xrange(period)]
	return (lookup_table[i%period] for i in count(0))

def square_wave(frequency=440.0, framerate=44100, amplitude=0.5):
	for s in sine_wave(frequency, framerate, amplitude):
		if s > 0:
			yield amplitude
		elif s < 0:
			yield -amplitude
		else:
			yield 0.0

def damped_wave(frequency=440.0, framerate=44100, amplitude=0.5, length=44100):
	if amplitude > 1.0: amplitude = 1.0
	if amplitude < 0.0: amplitude = 0.0
	return (math.exp(-(float(i%length)/float(framerate))) * s for i, s in enumerate(sine_wave(frequency, framerate, amplitude)))

def white_noise(amplitude=0.5):
	'''
	Generate random samples.
	'''
	return (float(amplitude) * random.uniform(-1, 1) for i in count(0))

def compute_samples(channels, nsamples=None):
	'''
	create a generator which computes the samples.

	essentially it creates a sequence of the sum of each function in the channel
	at each sample in the file for each channel.
	'''
	b = islice(izip(*(imap(sum, izip(*channel)) for channel in channels)), nsamples)
	return b 

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

def write_pcm(f, samples, sampwidth=2, framerate=44100, bufsize=2048):
    "Write samples as raw PCM data."
    max_amplitude = float(int((2 ** (sampwidth * 8)) / 2) - 1)

    # split the samples into chunks (to reduce memory consumption and improve performance)
    for chunk in grouper(bufsize, samples):
        frames = ''.join(''.join(struct.pack('h', int(max_amplitude * sample)) for sample in channels) for channels in chunk if channels is not None)
        f.write(frames)
    
    f.close()

    return filename

def binarize(word):
	'''
	Take in an ASCII-encoded string output it in binary

	Truncate the 0b specifier at the front
	'''
	return '0' + bin(int(binascii.hexlify(word), 16))[2:]

def getAndParseInput():
	'''
	Ask for input and return it as a list of binaries (each word separate)
	'''
	message = raw_input("What message to send? ")
	binary = binarize(message)
	return binary

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--channels', help="Number of channels to produce", default=2, type=int)
	parser.add_argument('-b', '--bits', help="Number of bits in each sample", choices=(16,), default=16, type=int)
	parser.add_argument('-r', '--rate', help="Sample rate in Hz", default=44100, type=int)
	parser.add_argument('-t', '--time', help="Duration of the wave in seconds.", default = 0.4, type=float)	
	parser.add_argument('-a', '--amplitude', help="Amplitude of the wave on a scale of 0.0-1.0", default=0.5, type=float)
	parser.add_argument('-f', '--frequency', help="Seed for frequencies of the wave in Hz", default=2000.0, type=float)
	parser.add_argument('-s', '--step', help="Step for generating next frequencies in Hz", default=97.0, type=float)	
	parser.add_argument('filename', help="The file to generate.", default="yes")
	args = parser.parse_args()
	#For some weird reason we produce a wave of double the frequency
	args.frequency = args.frequency/2
	#how many letters we send simultaneously / how many bits we send at once
	simultaneous = 1
	seed = args.frequency

	#Orthogonal frequencies. Notice that the receiver has to be seeded with the same values
	listOfFreqs = [1000, 2500]

	binary = getAndParseInput()

	#generate start signal
	channels = ((sine_wave(615, args.rate, args.amplitude),) for i in range(1))
	#length of start signal -> 1
	samples = compute_samples(channels, args.rate *0.5)
	#write and play
	write_wavefile("start", samples, args.rate * args.time, args.channels, args.bits / 8, args.rate)
	proc = Popen(["aplay", "start0"])
	
	#which file to write to / from which file to read
	n=0
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
			channels = ((sine_wave(listOfFreqs[(2*i)+int(subBinaryList[i])], args.rate, args.amplitude),) for i in range(len(subBinaryList)))
			samples = compute_samples(channels, args.rate * args.time)
			
			if args.filename == '-':
				filename = sys.stdout
			else:
				filename = args.filename
			write_wavefile(filename, samples, args.rate * args.time, args.channels, args.bits / 8, args.rate, n=n)
			
			if 'proc' in locals():
				proc.wait()
				time.sleep(0.5)
			
			proc = Popen(["aplay", filename + str(n)])
			n = (n + 1) % 2
	
	#generate stop signal
	channels = ((sine_wave(615, args.rate, args.amplitude),) for i in range(1))
	#length of stop signal -> 1
	samples = compute_samples(channels, args.rate * 0.5)
	#write and play
	write_wavefile("stop", samples, args.rate * args.time, args.channels, args.bits / 8, args.rate)
	if 'proc' in locals():
		proc.wait()
	proc = Popen(["aplay", "stop0"])
	
    #each channel is defined by infinite functions which are added to produce a sample.
#	channels = ((sine_wave(args.frequency, args.rate, args.amplitude),) for i in range(1))

    # convert the channel functions into waveforms
#	samples = compute_samples(channels, args.rate * 30)

    # write the samples to a file
#	if args.filename == '-':
#		filename = sys.stdout
#	else:
#		filename = args.filename
#	write_wavefile(filename, samples, args.rate * args.time, args.channels, args.bits / 8, args.rate)

if __name__ == "__main__":
    main()
#!/usr/bin/env python
import sys
