from receive_api import *
from generate_api import *
import time
import pdb

frequencyError = 150
rx = Receiver(printDebug = 1)
tx = Transmitter(duration = 0.3, amplitude = 1)

def synchronize():
	#we will repeat this until we die. Basically to synchro
	while True:
		#Make sure the receiver timed out
		time.sleep(tx.sleepTime * 10)
		#Send initial sync
		print "INITIAL SYNC"
		tx.sendSingleSignal(freq = tx.EOWFreq)
		#Wait for ack
		print "WAITING FOR ACK"
		p = rx.waitForSignal(frequency = tx.startFreq, errorAllowable = frequencyError)
		
		#We time out, try again but make sure they timed out by sleeping
		if p == 0:
			continue
		#Send in order: Sync, everything at once, sync
		time.sleep(tx.duration)
		print "SYNC"
		tx.sendSingleSignal(freq = tx.EOWFreq)
		print "ALL SIGNALS"
		tx.sendAllKnownSignals(duration = 0.5)
		print "SYNC"
		tx.sendSingleSignal(freq = tx.EOWFreq)
		
		#Wait for ack after they receive it
		print "WAIT FOR ACK FOR RECEIVED"
		p = rx.waitForSignal(frequency = tx.startFreq, errorAllowable = frequencyError)
		#Wait for all the acks they can give
		
		print "WAIT FOR ACK FOR EACH DECODED"
		counter = 0
		while rx.waitForSignal(frequency = tx.startFreq, errorAllowable = frequencyError):
			counter += 1
		print "Got ", counter, "Signals!"
		print "Should have gotten", 2 + len(tx.freqs), " signals: ", tx.freqs
		#Right number?
		if counter == 2 + len(tx.freqs):
			#And exit function
			print "Handshake complete"
			break
		else:
			#Again if no
			print "Retryin..."
			continue

#tx.sendSingleSignal(freq = tx.startFreq, duration = 1)
#tx.sendSingleSignal(freq = tx.freqs[0], duration = 1)
#tx.sendSingleSignal(freq = tx.freqs[1], duration = 1)
#tx.sendAllKnownSignals(duration = 1)
#tx.sendSingleSignal(freq = tx.EOWFreq, duration = 1)

synchronize()
