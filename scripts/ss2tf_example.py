import control
import numpy as np
import matplotlib.pyplot as plt

m1 = 1
m2 = 2
b1 = 1
b2 = 1
k1 = 1
k2 = 2

input = 0
output = 0

A = np.array([[0,0,1,0],
            [0, 0, 0, 1], 
            [-(k1+k2)/m1, k2/m1, -(b1+b2)/m1, b2/m1],
            [k2/m2, -k2/m2, b2/m2, -b2/m2]])

B = np.array([[0, 0],
              [0, 0],
              [1/m1, 0],
              [0, 1/m2]])

C = np.array([[1,0,0,0],
              [0,1,0,0]])

D = np.array([[0,0],[0,0]])

tf = control.ss2tf(A, B[:,input], C[output], D[output,input])

ss = control.StateSpace(A, B[:,input], C[output], D[output,input])

freq, mag, phase = control.bode(ss)


plt.show()