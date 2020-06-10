import numpy
import matplotlib.pyplot as plt

def sigmoid(x):
    y= 1/(1+numpy.e**-x)
    return y

t1=[]
for i in range(-4,11):
    t1.append(i)
t1=numpy.array(t1)

t2 = 2*t1+numpy.random.normal(0,2,(15,))


data = numpy.array([t1,t2])

#data=numpy.array([[-1,-1,0,2,0],[-2,0,0,1,1]])
plt.scatter(data[0],data[1],marker='^',c='blue')


cov = data.dot(data.T)

val,vec = numpy.linalg.eigh(cov.T)

axes=vec[1]


axes=axes.reshape(1,2)

recover = axes.T.dot(axes).dot(data)
plt.scatter(recover[0],recover[1],marker='o',c='red')

plt.show()
           
