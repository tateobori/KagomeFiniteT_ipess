import numpy as np
import matplotlib.pyplot as plt
import argparse


def LoadingData(filename):

	# Loading the data
	beta, E = np.loadtxt(str(filename), delimiter=',', unpack=True)

	# Remove the first temperature data to match the shaoe of specific heat
	beta = np.delete(beta, 0)
	T = 1.0/beta/0.0861734

	# Calculate the specific heat from the Energy and beta
	specific_heat = -beta*beta*np.diff(E, n=1)/0.001*96.4855348

	return T, specific_heat, specific_heat/T


if __name__=="__main__":

	parser = argparse.ArgumentParser(description='',allow_abbrev=False)
	parser.add_argument("--file", default="output", help="Input DataSet File")
	parser.add_argument("--xlim", type=float, default=3.0, help="xlim to show")
	args = parser.parse_args()
	file = args.file
	T1, C1, S1 = LoadingData(file)

	fig = plt.figure()
	plt.title("Specific Heat of Quantum Ising on Kagome Lattice")
	plt.xlabel("T")
	plt.ylabel("C")
	plt.xlim(0, args.xlim)
	plt.ylim(0, 12)
	plt.grid() 

	plt.plot(T1, C1, color="black", label="D10")
	plt.legend()
	fig.savefig("test.png")
	plt.show()
	plt.close()

	fig = plt.figure()
	plt.title("Specific Heat of Quantum Ising on Kagome Lattice")
	plt.xlabel("T")
	plt.ylabel("C/T")
	plt.xlim(0, args.xlim)
	plt.ylim(0, 12)
	plt.grid() 

	plt.plot(T1, S1, color="black", label="D10")
	plt.legend()
	fig.savefig("test1.png")
	plt.show()

