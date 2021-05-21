# -*- coding: utf-8 -*-

import sys
import time
import torch
import argparse
from scipy.linalg import expm
import scipy.linalg as spl
import numpy as np
from msvd import tensor_svd
from msvd import tensor_eigh
from msvd import tensor_QR
from msvd import psvd
import itertools
from itertools import product
from collections import OrderedDict
import json

"""
クラス Tensors_CTMに格納されているテンソルの定義


    C1--1     0--T1--2     0--C2
    |            |             |
    0            1             1

    2            0             0          
    |            |             |          
    T4--1     3--A--1     1 --T2       
    |            |             |        
    0            2             2          

    1           1              0
    |           |              | 
    C4--0    2--T3--0      1--C3

PEPSの順番
          0          0            0        0
         /          /             |        |
    3-- a --1   3-- b --1      3--A--1  3--B--1
      / |         / |             |        |
     2  4        2  4             2        2

Isometryの引数の定義

C1 -1       0--T11--2       0--T12--2       0- C2 
|               |1              |1              |
0                                               1

2                                               0
|                                               |
T42-1           A1            A2             1-T21
|                                               |
0                                               2

2                                               0
|                                               |    
T41-1           A4            A3            1-T22
|                                               |
0                                               2
 
1                                               0
|               |1          |1                  |
C4 -0       2--T32--0    2--T31--0           1- C3

  0    1  2
  |    |  |
  P    P_til
 | |    |
 1 2
"""

################################################################
def spin_operators(S):

    d = int(np.rint(2*S + 1))
    dz = np.zeros(d);  mp = np.zeros(d-1)

    for n in range(d-1):
        dz[n] = S - n
        mp[n] = np.sqrt((2.0*S - n)*(n + 1.0))

        dz[d - 1] = - S
    Sp = np.diag(mp,1);   Sm = np.diag(mp,-1)
    Sx = 0.5*(Sp + Sm);   Sy = -0.5j*(Sp - Sm)
    Sz = np.diag(dz)


    return Sx, Sy, Sz

def Hamiltonian_Heisen_In_Trian(J,Hz,spin):

    Sx, Sy, Sz = spin_operators(spin)
    I =np.eye(d_spin,d_spin)

    H_BC = np.kron(I, np.kron(Sx,Sx)) + np.kron(I, np.kron(Sy,Sy)) + np.kron(I, np.kron(Sz,Sz))
    H_AB = np.kron(np.kron(Sx,Sx), I) + np.kron(np.kron(Sy,Sy), I) + np.kron(np.kron(Sz,Sz), I)
    H_CA = np.kron(np.kron(Sx,I), Sx) + np.kron(np.kron(Sy,I), Sy) + np.kron(np.kron(Sz,I), Sz)

    Ham = J*(np.kron(Sx,Sx) + np.kron(Sy,Sy) + np.kron(Sz,Sz)) - 0.25 * Hz *( np.kron(Sz,I) + np.kron(I,Sz) )
    H =  J*(H_AB + H_BC + H_CA) - 0.5*Hz*(np.kron(np.kron(Sz,I), I) + np.kron(np.kron(I,Sz), I) + np.kron(np.kron(I,I), Sz))
    #print(np.real(H_AB))
    #print(np.real(H_BC))
    #print(np.real(H_CA))
    #print(np.real(H))
    #exit()



    return np.real(H), np.real(Ham.reshape(d_spin, d_spin, d_spin, d_spin))

def Hamiltonian_Ising_In_Trian(J,Hx,spin):

    Sx, Sy, Sz = spin_operators(spin)
    I =np.eye(d_spin,d_spin)

    H_BC = np.kron(I, np.kron(Sz,Sz))
    H_AB = np.kron(np.kron(Sz,Sz), I)
    H_CA = np.kron(np.kron(Sz,I), Sz)

    Ham = J*(np.kron(Sz,Sz)) - 0.25 * Hx *( np.kron(Sx,I) + np.kron(I,Sx) )
    H =  J*(H_AB + H_BC + H_CA) - 0.5*Hx*(np.kron(np.kron(Sx,I), I) + np.kron(np.kron(I,Sx), I) + np.kron(np.kron(I,I), Sx))
    #print(np.real(H_AB))
    #print(np.real(H_BC))
    #print(np.real(H_CA))
    #print(np.real(H))
    #exit()



    return np.real(H), np.real(Ham.reshape(d_spin, d_spin, d_spin, d_spin))
###########################################################################
def initial_iPESS(d_spin):

    ## High temperature limit
    
    A1 = np.zeros((1, d_spin, 1, d_spin))
    A1[0,0,0,0]=A1[0,1,0,1]=1.0
    B1 = C1 = A1

    R_up  = np.ones((1,1,1))# + 1.0j
    R_low = np.ones((1,1,1))# + 1.0j

    # vector lu, lr, ld, ll
    l = np.ones(A1.shape[0], dtype=float)
    for i in np.arange(len(l)):    l[i] /= 10**i
    l /= np.sqrt(np.dot(l,l))
    
    return A1, B1, C1, R_up, R_low, l,l,l,l,l,l

def SimpleUpdate_down(A,B,C,R,la,lb,lc,U):
    #
    #             0 1
    #  \|     |/   \|
    #   A     B     A     index 0: outgoing
    #    \ | /       \2   index 2: ingoint
    #      R
    #      |          0   1
    #      C/          \R/
    #      |            |
    #                   2
    #
    #

    A = A*la[:,None,None,None] 
    B = B*lb[:,None,None,None]
    C = C*lc[:,None,None,None]

    
    T = np.transpose(
        np.tensordot(
            A, np.tensordot(
                B, np.tensordot(
                    C, R, ([2], [2])
                ), ([2], [4])
            ), ([2], [6])
        ), [1, 4, 7, 2, 5, 8, 0, 3, 6]
    )


    V  = np.tensordot(
        U, T, ([3, 4, 5], [0, 1, 2])
    )


    Tmp = np.tensordot(V, V.conj(),([1,2,4,5,7,8], [1,2,4,5,7,8]) ) ##  (0,3,6)
    #uA, la_new = tensor_eigh(Tmp, (0,1),(2,3),D)
    uA, la_new, _ = tensor_svd(Tmp,(0,1,2),(3,4,5),D)
    la_new = np.sqrt(la_new)
    la_new = la_new/np.sqrt(np.dot(la_new,la_new))

    la_new = la_new[~(la_new < 1e-8)]
    uA = uA[:,:,:,:la_new.shape[0]]

    A = uA*(1/la)[None,None,:,None]
    A = A.transpose(2,0,3,1)

    Tmp = np.tensordot(V, V.conj(),([0,2,3,5,6,8], [0,2,3,5,6,8]) ) ##  (1,4,7)
    #uB, lb_new = tensor_eigh(Tmp, (0,1),(2,3),D)
    uB, lb_new, _ = tensor_svd(Tmp,(0,1,2),(3,4,5),D)
    lb_new = np.sqrt(lb_new)
    lb_new = lb_new/np.sqrt(np.dot(lb_new,lb_new))

    lb_new = lb_new[~(lb_new < 1e-8)]
    uB = uB[:,:,:,:lb_new.shape[0]]

    B = uB*(1/lb)[None,None,:,None] 
    B = B.transpose(2,0,3,1)

    Tmp = np.tensordot(V, V.conj(),([0,1,3,4,6,7], [0,1,3,4,6,7]) ) ##  (2,5,8)
    #uC, lc_new = tensor_eigh(Tmp, (0,1),(2,3),D)
    uC, lc_new, _ = tensor_svd(Tmp,(0,1,2),(3,4,5),D)
    lc_new = np.sqrt(lc_new)
    lc_new = lc_new/np.sqrt(np.dot(lc_new,lc_new))

    lc_new = lc_new[~(lc_new < 1e-8)]
    uC = uC[:,:,:,:lc_new.shape[0]]

    C = uC*(1/lc)[None,None,:,None] 
    C = C.transpose(2,0,3,1)
 
    R_new = np.tensordot(
        uA, np.tensordot(
            uB, np.tensordot(
                V, uC, ([2, 5, 8], [0, 1, 2])
            ), ([0, 1, 2], [1, 3, 5])
        ), ([0, 1, 2], [1, 2, 3])
    )

    R_new /=np.max(abs(R_new))

  


    return A, B, C, R_new, la_new, lb_new, lc_new

def SimpleUpdate_up(B,C,A,R,lb,lc,la,U):
    #
 
    B =B*lb[None,None,:,None]
    C =C*lc[None,None,:,None]
    A =A*la[None,None,:,None]

    T = np.transpose(
        np.tensordot(
            B, np.tensordot(
                C, np.tensordot(
                    A, R, ([0], [2])
                ), ([0], [4])
            ), ([0], [6])
        ), [0, 3, 6, 2, 5, 8, 1, 4, 7]
    )

    V  = np.tensordot(
        U, T, ([3, 4, 5], [0, 1, 2])
    )

    Tmp = np.tensordot(V, V.conj(),([1,2,4,5,7,8], [1,2,4,5,7,8]) ) ##  (0,3,6)
    uB, lb_new, _ = tensor_svd(Tmp,(0,1,2),(3,4,5),D)
    lb_new = np.sqrt(lb_new)
    lb_new = lb_new/np.sqrt(np.dot(lb_new,lb_new))

    lb_new = lb_new[~(lb_new < args.eps_TEBD)]
    uB = uB[:,:,:,:lb_new.shape[0]]
    B = np.transpose(uB*(1/lb)[None,None,:,None],[3,0,2,1])

    Tmp = np.tensordot(V, V.conj(),([0,2,3,5,6,8], [0,2,3,5,6,8]) ) ##  (1,4,7)
    uC, lc_new, _ = tensor_svd(Tmp,(0,1,2),(3,4,5),D)
    lc_new = np.sqrt(lc_new)
    lc_new = lc_new/np.sqrt(np.dot(lc_new,lc_new))

    lc_new = lc_new[~(lc_new < args.eps_TEBD)]
    uC = uC[:,:,:,:lc_new.shape[0]]
    C = np.transpose(uC*(1/lc)[None,None,:,None],[3,0,2,1]) 

    Tmp = np.tensordot(V, V.conj(),([0,1,3,4,6,7], [0,1,3,4,6,7]) ) ##  (2,5,8)
    uA, la_new, _ = tensor_svd(Tmp,(0,1,2),(3,4,5),D)
    la_new = np.sqrt(la_new)
    la_new = la_new/np.sqrt(np.dot(la_new,la_new))

    la_new = la_new[~(la_new < args.eps_TEBD)]
    uA = uA[:,:,:,:la_new.shape[0]]
    A = np.transpose(uA*(1/la)[None,None,:,None] ,[3,0,2,1])

    R_new = np.tensordot(
        uB.conj(), np.tensordot(
            uC.conj(), np.tensordot(
                uA.conj(), V, ([0, 1, 2], [2, 5, 8])
            ), ([0, 1, 2], [2, 4, 6])
        ), ([0, 1, 2], [2, 3, 4])
    )
    R_new /=np.max(abs(R_new))



    return B, C, A, R_new, lb_new, lc_new, la_new

###########################################################################

def Energy_Triangle_down(A,B,C,R, la_up, lb_up, lc_up, H, Ham):

    A = A*la_up[:,None,None,None] ; B = B*lb_up[:,None,None,None]; C = C*lc_up[:,None,None,None]
    H = H.reshape(d_spin, d_spin, d_spin, d_spin, d_spin, d_spin)


    # ((A*Adag)*((R*(B*Bdag))*(Rdag*(C*Cdag))))
    tmp = np.transpose(
        np.tensordot(
            np.tensordot(
                A, A, ([0, 3], [0, 3])
            ), np.tensordot(
                np.tensordot(
                    R, np.tensordot(
                        B, B, ([0, 3], [0, 3])
                    ), ([1], [1])
                ), np.tensordot(
                    R, np.tensordot(
                        C, C, ([0, 3], [0, 3])
                    ), ([2], [3])
                ), ([1, 4], [3, 1])
            ), ([1, 3], [0, 3])
        ), [0, 2, 4, 1, 3, 5]
    )
    
    norm = np.einsum(tmp, (0, 1, 2, 0, 1, 2), ())

    E_AB = np.einsum(tmp, (0, 1, 2, 3, 4, 2), (0, 1, 3, 4)) 
    E_AB = np.tensordot(E_AB, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E_BC = np.einsum(tmp, (0, 1, 2, 0, 3, 4), (1, 2, 3, 4))
    E_BC = np.tensordot(E_BC, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E_CA = np.einsum(tmp, (0, 1, 2, 3, 1, 4), (0, 2, 3, 4))
    E_CA = np.tensordot(E_CA, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E = np.tensordot(tmp.conj(), H, ([0,1,2,3,4,5],[0,1,2,3,4,5]))/norm


    return np.real(E), np.real(E_AB), np.real(E_BC), np.real(E_CA)

def Energy_Triangle_up(B,C,A,R, lb_low, lc_low, la_low, H, Ham):

    B = B*lb_low[None,None,:,None] ; C = C*lc_low[None,None,:,None]; A = A*la_low[None,None,:,None]
    H = H.reshape(d_spin, d_spin, d_spin, d_spin, d_spin, d_spin)


    # ((B*B)*((R*(C*C))*(R*(A*A))))
    tmp = np.transpose(
        np.tensordot(
            np.tensordot(
                B, B, ([2, 3], [2, 3])
            ), np.tensordot(
                np.tensordot(
                    R, np.tensordot(
                        C, C, ([2, 3], [2, 3])
                    ), ([1], [0])
                ), np.tensordot(
                    R, np.tensordot(
                        A, A, ([2, 3], [2, 3])
                    ), ([2], [2])
                ), ([1, 3], [2, 1])
            ), ([0, 2], [0, 3])
        ), [0, 2, 4, 1, 3, 5]
    )
    
    norm = np.einsum(tmp, (0, 1, 2, 0, 1, 2), ())

    E_BC = np.einsum(tmp, (0, 1, 2, 3, 4, 2), (0, 1, 3, 4)) 
    E_BC = np.tensordot(E_BC, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E_CA = np.einsum(tmp, (0, 1, 2, 0, 3, 4), (1, 2, 3, 4))
    E_CA = np.tensordot(E_CA, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E_AB = np.einsum(tmp, (0, 1, 2, 3, 1, 4), (0, 2, 3, 4))
    E_AB = np.tensordot(E_AB, Ham, ([0,1,2,3],[0,1,2,3]))/norm

    E = np.tensordot(tmp.conj(), H, ([0,1,2,3,4,5],[0,1,2,3,4,5]))/norm


    return np.real(E), np.real(E_BC), np.real(E_CA), np.real(E_AB)

def Magnetization(A, la_up, la_low):

    Sx,Sy,Sz=spin_operators(spin)

    I = np.eye(d_spin, d_spin)
    A = A*la_up[:,None,None]*la_low[None,None,:]

    mz = np.tensordot(
        Sz, np.tensordot(
            A, A.conj(), ([0, 2], [0, 2])
        ), ([0, 1], [0, 1])
    )

    my = np.tensordot(
        Sy, np.tensordot(
            A, A.conj(), ([0, 2], [0, 2])
        ), ([0, 1], [0, 1])
    )

    mx = np.tensordot(
        Sx, np.tensordot(
            A, A.conj(), ([0, 2], [0, 2])
        ), ([0, 1], [0, 1])
    )




    return mx, my, mz

###########################################################################
def Calcu_Unit_down(A,B,C,R_up,R_low):

        psi = np.transpose(
            np.tensordot(
                np.tensordot(
                    C, R_up, ([0], [1])
                ), np.tensordot(
                    A, np.tensordot(
                        B, R_low, ([2], [1])
                    ), ([2], [2])
                ), ([1], [4])
            ), [4, 6, 0, 3, 1, 2, 5]
        )

        return psi.reshape(d_spin**3,D,D,D,D)

###########################################################################
if __name__=="__main__":
    # obtain the arguments

    parser = argparse.ArgumentParser(description='',allow_abbrev=False)
    #parser.add_argument("--omp_cores", type=int, default=1,help="number of OpenMP cores")
    parser.add_argument("--D", type=int, default=2, help="Virtual bond dimension")
    parser.add_argument("--J", type=float, default=1, help="coupling constant on down trianle")
    parser.add_argument("--J_up", type=float, default=1, help="coupling constant on up triangle")
    parser.add_argument("--dt", type=float, default=0.01, help="inmaginary time")
    parser.add_argument("--chi", type=int, default=20, help="bond dimensions of CTM")
    parser.add_argument("--spin", type=float, default=0.5, help="spin value")
    parser.add_argument("--Hz_start", type=float, default=0., help="intiail value of magnetic field")
    parser.add_argument("--Hz_end", type=float, default=3.0, help="final value of magnetic field")
    parser.add_argument("--Hz_step", type=float, default=0.1, help="step of the magnetic field")
    parser.add_argument("--maxstepTEBD", type=int, default=10000000, help="maximal number of TEBD iterations")
    parser.add_argument("--maxstepCTM", type=int, default=10, help="maximal number ofCTM iterations")
    parser.add_argument("--eps_TEBD", type=float, default=1e-9, help="TEBD criterion for convergence")
    parser.add_argument("--instate", default="test", help="Input state JSON")
    parser.add_argument("--beta_end", type=float, default=4.00, help="final beta value to calculate")
    parser.add_argument("--Hx", type=float, default=0.50, help="transverse field")

    args = parser.parse_args()

    D= args.D
    J= args.J
    J_up= args.J_up
    dt= args.dt
    chi= args.chi
    spin = args.spin
    Hz_start= args.Hz_start
    Hz_end= args.Hz_end
    Hz_step= args.Hz_step
    maxstepTEBD= args.maxstepTEBD
    maxstepCTM= args.maxstepCTM
    d_spin = int(2*spin + 1 )
    tau = dt
    temp = 0.0
    Hx = args.Hx

    # criterion for convergence
    eps_TEBD = 1e-9;  eps_CTM = 10**(-10)
    
    # intiail iPESS
    A1, B1, C1,\
    R1_up, R1_low,\
    l_A1_up, l_B1_up, l_C1_up,\
    l_A1_low, l_B1_low, l_C1_low =initial_iPESS(d_spin)

    # open the text file
    name = args.instate
    f = open(name+'.txt','w')

    for Hz in np.arange(Hz_start, Hz_end, Hz_step):

        for i in range(maxstepTEBD):

            H, Ham = Hamiltonian_Heisen_In_Trian(J,Hz,spin)
            H1, Ham1 = Hamiltonian_Heisen_In_Trian(J_up,Hz,spin)
            #H, Ham = Hamiltonian_Ising_In_Trian(J,Hx,spin)
            #H1, Ham1 = Hamiltonian_Ising_In_Trian(J,Hx,spin)

            U = expm(-0.5*dt*H).reshape(d_spin, d_spin,d_spin, d_spin, d_spin, d_spin)
            U1 = expm(-0.5*dt*H1).reshape(d_spin, d_spin,d_spin, d_spin, d_spin, d_spin)

            
            A1, B1, C1, R1_low, l_A1_low, l_B1_low, l_C1_low = \
            SimpleUpdate_down(A1,B1,C1,R1_low,l_A1_up,l_B1_up,l_C1_up,U)
                    
            B1, C1, A1, R1_up, l_B1_up, l_C1_up, l_A1_up = \
            SimpleUpdate_up(B1,C1,A1,R1_up,l_B1_low,l_C1_low,l_A1_low,U1)

            E0, E0_AB, E0_BC, E0_CA = Energy_Triangle_down(A1, B1, C1, R1_low, l_A1_up, l_B1_up, l_C1_up, H, Ham)
            E1, E1_BC, E1_CA, E1_AB = Energy_Triangle_up(B1, C1, A1, R1_up, l_B1_low, l_C1_low, l_A1_low, H1, Ham1)

            if 1.0/((i+1)*dt)<=3.0:
                f.write("{0:.4e}, {1:.9e}\n".format((i+1)*dt, (E0+E1)/3.0))
            if i%100 ==0:
                print("{0:.2e}, {1:.4e}, {2:.4e}, {3:.4e}, {4:.9e}".format(i*dt, E0_AB, E0_BC, E0_CA, (E0+E1)/3.0))
                print("{0:.2e}, {1:.4e}, {2:.4e}, {3:.4e}, {4:.9e},\n".format(i*dt, E1_AB, E1_BC, E1_CA, (E0+E1)/3.0))
                #print(l_A1_low)
                #print(l_B1_low)
                #print(l_C1_low,"\n")

            if i*dt>args.beta_end:
                break 

        break
        



        
        
        
    
    

        
        
        
            
            
          

























