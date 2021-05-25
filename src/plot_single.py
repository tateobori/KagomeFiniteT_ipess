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

	return T, specific_heat, specific_heat/T


if __name__=="__main__":

    # obtain the arguments
	T1, C1, v1 = LoadingData("ising-D6-Hx0025.txt")
	T2, C2, v2 = LoadingData("ising-D6-Hx005.txt")
	T3, C3, v3 = LoadingData("ising-D6-Hx025.txt")
	T4, C4, v4 = LoadingData("ising-D6-Hx050.txt")


	fig = plt.figure()
	c1,c2,c3,c4 = "blue","green","red","black"      # 各プロットの色
	l1,l2,l3,l4 = "0.025","0.050","0.25","0.50"   # 各ラベル

	plt.title("Specific Heat of Quantum Ising on Kagome Lattice")
	plt.xlabel("T")
	plt.ylabel("C")
	plt.xlim(0,0.8)
	plt.grid() 

	plt.plot(T1, v1, color=c1, label=l1)
	plt.plot(T2, v2, color=c2, label=l2)
	plt.plot(T3, v3, color=c3, label=l3)
	plt.plot(T4, v4, color=c4, label=l4)
	plt.legend()
	fig.savefig("ising-cdivt.png")
	plt.show()


