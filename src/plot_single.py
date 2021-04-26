import numpy as np
import matplotlib.pyplot as plt


def LoadingData(filename):

	# Loading the data
	beta, E = np.loadtxt(str(filename), delimiter=',', unpack=True)

	# Remove the first temperature data to match the shaoe of specific heat
	beta = np.delete(beta, 0)
	T = 1.0/beta

	# Calculate the specific heat from the Energy and beta
	specific_heat = -beta*beta*np.diff(E, n=1)/0.001

	return T, specific_heat


if __name__=="__main__":

	T1, C1 = LoadingData("output.txt")
	T2, C2 = LoadingData("outputD3.txt")
	T3, C3 = LoadingData("outputD4.txt")


	fig = plt.figure()
	c1,c2,c3 = "blue","green","red"      # 各プロットの色
	l1,l2,l3 = "D2","D3","D4"   # 各ラベル

	plt.title("Specific Heat of Quantum Ising on Kagome Lattice")
	plt.xlabel("T")
	plt.ylabel("C")
	plt.xlim(0,0.8)
	plt.grid() 

	plt.plot(T1, C1, color=c1, label=l1)
	plt.plot(T2, C2, color=c2, label=l2)
	plt.plot(T3, C3, color=c3, label=l3)
	plt.legend()
	fig.savefig("img.png")
	plt.show()


