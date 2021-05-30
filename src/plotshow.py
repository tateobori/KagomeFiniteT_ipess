import numpy as np
import matplotlib.pyplot as plt
import sys


def LoadingData(filename):

	# Loading the data
	beta, E = np.loadtxt(str(filename), delimiter=',', unpack=True)

	# Remove the first temperature data to match the shaoe of specific heat
	beta = np.delete(beta, 0)
	T = 1.0/beta

	# Calculate the specific heat from the Energy and beta
	specific_heat = -beta*beta*np.diff(E, n=1)/0.001

	return T, specific_heat, specific_heat/T


if __name__=="__main__":


	fig = plt.figure()
	plt.title("Specific Heat of Quantum Breathing Kagome Model")
	plt.xlabel("T")
	plt.ylabel("C")
	plt.xlim(0, 0.30)
	plt.grid() 

	j=0
	arg = sys.argv
	del arg[0]
	for file in sys.argv:
		T1, C1, C2 = LoadingData(file)
		#plt.plot(T1, C1, label=str(file))
		plt.plot(T1, C2, label=str(file))
		j +=1

	plt.legend()
	fig.savefig("test1.png")
	plt.show()


