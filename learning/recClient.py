from receive_api import *
from generate_api import *
import time

frequencyError = 100
rx = Receiver(printDebug = 1)
tx = Transmitter(duration = 0.3, amplitude = 1)

def synchronize():
	rx.purgeFreqs()
	#Send ack
	print "INITIAL ACK"
	tx.sendSingleSignal(freq = tx.startFreq)
	#Get sync
	p = rx.waitForSignal(frequency = tx.EOWFreq, errorAllowable = frequencyError)
	
	#We time out, return to snooping
	if p == 0:
		return
	#Accept all incoming signal
	rx.recordInput()
	#Until we get another sync
	p = rx.waitForSignal(frequency = tx.EOWFreq, errorAllowable = frequencyError, purge = 0)
	
	#Timeout reached!
	if p == 0:
		return
	rx.stopRecordingInput()
	#Send Ack
	tx.sendSingleSignal(freq = tx.startFreq)
	
	#Decode the freqs
	freqsDecoded = rx.getFreqs(tx.EOWFreq, tx.startFreq, tx.freqs)
	#Send acks = the number of freqs decoded
	for i in range(freqsDecoded):
		tx.sendSingleSignal(freq = tx.startFreq)

while(True):
	snoopResponse = (0,0)

	while snoopResponse[0] == 0:
		snoopResponse = rx.snoop(timeOut = 10000)
	
#	if np.abs(tx.startFreq - snoopResponse[1]) < frequencyError:
#		rx.waitForMessage()
	if np.abs(tx.EOWFreq - snoopResponse[1]) < frequencyError:
		synchronize()


