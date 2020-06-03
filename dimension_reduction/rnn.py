# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:17:04 2020
@author: admin
"""
import numpy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from queue import  PriorityQueue as p



n=1000
def load_data(n):
    swiss_roll =datasets.make_swiss_roll(n_samples=n)
    return swiss_roll[0],np.floor(swiss_roll[1])

X,Y=load_data(n)

def cal_distance(data):
    sample = len(data)
    matrix = numpy.zeros((sample,sample))
    for i in range(sample):
        line = data[i]
        rest = data-line
        rest = rest**2
        s = numpy.sum(rest,1)
        s=numpy.sqrt(s)
        matrix[i]=s
    return matrix


def floyd(m):
    l=len(m)
    for p in range(l):
        for i in range(l):
            for j in range(l):
                if m[i,j]>m[i,p]+m[p,j]:
                    m[i,j]=m[i,p]+m[p,j]
    return m


def neighbour(matrix,k):
    n = matrix.shape[0]
    nei = numpy.ones((n,k))
    for i in range(0,n):
        que = p()
        temp = numpy.zeros(k, )
        line = matrix[i]
        for j in range(0,n):
            que.put((line[j],j))
        que.get()
        for z in range(0,k):
            item = que.get()
            temp[z]=item[1]
        nei[i]=temp
    nei=numpy.int16(nei)
    return nei



mat = cal_distance(X)
k=8
nn=neighbour(mat,k)

dm=numpy.ones((n,n))*3000
for i in range(0,n):
    for j in range(0,k):
        x=nn[i][j]
        dm[i,x]=mat[i,x]
        


mat=floyd(dm)

del dm
m2 = mat**2
v = numpy.sum(m2)/(n*n)

def di(m):
    s=len(m)
    line=numpy.sum(m,1)/s
    return line


def dj(m):
    s=len(m)
    line=numpy.sum(m,0)/s
    return line

dist_i=di(m2)
dist_j=dj(m2)

b=numpy.zeros((n,n))

for i in range(0,n):
    for j in range(0,n):
        b[i,j]=-(m2[i,j]-dist_i[i]-dist_j[j]+v)/2



value,vector=numpy.linalg.eigh(b)
vector=vector.T
value=value[n-2:n]
vector=vector[n-2:n]


def iden(x):
    l=len(x)
    m=numpy.eye(l)
    for i in range(l):
        m[i,i]=x[i]
    return m


diag=numpy.eye(2)
diag[0,0]=value[0]
diag[1,1]=value[1]
diag=diag**0.5

reduce = diag.dot(vector)
reduce=reduce.T



plt.scatter(reduce[:,0],reduce[:,1],c=Y)
plt.show()

fig = plt.figure('data')
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker=',',c=Y)
plt.show()
























