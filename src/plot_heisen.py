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

	T1, C1 = LoadingData("finiteD8.txt")
	T2, C2 = LoadingData("finiteD10.txt")
	T3, C3 = LoadingData("finiteD12.txt")
	T4, C4 = LoadingData("finiteD14.txt")
	T5, C5 = LoadingData("HeisenD20.txt")


	fig = plt.figure()
	c1,c2,c3,c4,c5 = "blue","green","red","black","purple"      # 各プロットの色
	l1,l2,l3,l4,l5 = "D8","D10","D12","D14","D20"   # 各ラベル

	plt.title("Specific Heat of Quantum Ising on Kagome Lattice")
	plt.xlabel("T")
	plt.ylabel("C")
	plt.xscale('log')
	#plt.xlim(0,1.0)
	plt.grid() 

	plt.plot(T1, C1, color=c1, label=l1)
	plt.plot(T2, C2, color=c2, label=l2)
	plt.plot(T3, C3, color=c3, label=l3)
	plt.plot(T4, C4, color=c4, label=l4)
	plt.plot(T5, C5, color=c5, label=l5)
	plt.legend()
	fig.savefig("heisen.png")
	plt.show()


