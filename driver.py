import matplotlib.pyplot as plt
import numpy as np
from computeCost import computeCost
from gradientDescent import gradientDescent

path = r"C:\Users\Archit Mishra\Desktop\machine-learning-ex1\ex1\ex1data1.txt"
f = open(path,"r")
z = f.read()
c = z.split()

a,b = np.zeros((len(c),1)),np.zeros((len(c),1))
for i in range(len(c)):
    x = c[i].split(",")
    a[i] = x[0]          # a = X                numpy array
    b[i] = x[1]          # y                    numpy array

a = np.insert(a,0,values=1,axis=1)

plt.scatter(a[:,1:],b)
plt.show()

theta = np.zeros((2,1))
alpha = 0.01
iterations = 1500

print(computeCost(a,b,theta,alpha,iterations))

final_theta, j_hist = gradientDescent(a,b,theta,alpha,iterations)
print(final_theta)
x = [i for i in range(1500)]

plt.scatter(x,j_hist)
plt.show()