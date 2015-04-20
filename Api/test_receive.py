import receive_api

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--frequency', help="Seed for frequencies of the wave in Hz", default = 1000, type=float)
	parser.add_argument('-s', '--step', help="Step for generating next frequencies in Hz", default = 97.0, type=float)

	args = parser.parse_args()
	n = Receiver()
	n.start()

if __name__ == "__main__":
	main()
