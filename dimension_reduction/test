import numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_digits
from queue import  PriorityQueue as p
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from mpl_toolkits.mplot3d import Axes3D
digits = load_digits()
#data = digits.data
#pca=PCA(n_components=30)
#reduced = pca.fit_transform(digits.data)
color=['black','bisque','green','blue',
    'purple','yellow','aqua','gold'
      , 'crimson','azure']

def dist(a,b):
    temp = a -b
    temp = temp*temp
    return numpy.sqrt(numpy.sum(temp))

def load_data():
    swiss_roll =datasets.make_swiss_roll(n_samples=2000)
    return swiss_roll[0],np.floor(swiss_roll[1])


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


def build_gram(nei,data,i):
    line = list(numpy.int16(nei[i]))
    k=nei.shape[1]
    feature = data.shape[1]
    z1,z2 = numpy.ones((k,feature)),numpy.ones((k,feature))
    for t in range(0,k):
        z1[t]=data[line[t]]
        z2[t]=data[i]
    temp = z2-z1
    gram = temp.dot(temp.T)
    return gram



def cal_weight(gram):
    k=gram.shape[0]
    one = numpy.ones((k,1))
    try:

        weight = numpy.linalg.inv(gram).dot(one)
    except:
        co = numpy.trace(gram)/10000
        gram=gram+numpy.eye(k)*co
        #make it non singular
        weight = numpy.linalg.inv(gram).dot(one)
    s = float(sum(weight))
    weight = weight/s
    return  weight

X,Y=load_data()
data = X
mat = cal_distance(data)
kn=32
nei= neighbour(mat,kn)




def build_w(nei,data,k=kn):
    n,n2 = data.shape[0],data.shape[1]
    w= numpy.zeros((n,n))
    for i in range(n):
        gram=build_gram(nei,data,i)
        weight=cal_weight(gram)
        for t in range(k):
            w[i][nei[i][t]]=weight[t]
    return w


w = build_w(nei,data,kn)


def tsne(w):
    n=w.shape[0]
    I_y  = numpy.eye(n)
    W_y=w
    M = np.dot((I_y - W_y).T, (I_y - W_y))

    eig_val, eig_vector = np.linalg.eig(M)
    index_ = np.argsort(np.abs(eig_val))[1:2 + 1]

    print("index_", index_)
    print(index_)
    Y = eig_vector[:, index_]
    return Y


cor=tsne(w)
plt.subplot(121)
plt.scatter(cor[:,0],cor[:,1],c=Y)

plt.subplot(122)
fig = plt.figure('data')
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],marker=',',c=Y)
plt.show()

















'''
fig = plt.figure()
ax = Axes3D(fig)
for i in range(0,1500):
    vec = reduced[i]
    ax.scatter([vec[0]],[vec[1]],c=color[digits.target[i]])
plt.show()
'''

