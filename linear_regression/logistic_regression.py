import numpy
import matplotlib.pyplot as plt
x=[0,0,2,2]
y=[0,2,0,2]
plt.scatter(x,y)
x1=[4,4,6,6]
y1=[4,6,4,6]
plt.scatter(x1,y1)

x_data = numpy.array([[0,0,2,2,4,4,6,6],
                      [0,2,0,2,4,6,4,6]])
y_data=numpy.array([0,0,0,0,1,1,1,1])
lr = 0.01
ancilla = numpy.ones((1,8))
x_data = numpy.concatenate((ancilla,x_data),0)
weight = numpy.random.uniform(0,5,(3,))

def sigmoid(x):
    return 1/(1+numpy.e**(-x))



def loss(w,x,y):
    loss=numpy.zeros(w.shape)
    for p in range(3):
        temp = y*x[p]
        loss[p]=loss[p]+numpy.sum(temp)


    for t in range(8):
        p1 = sigmoid( numpy.sum(w*x[0:3,t]))
        for q in range(3):
            loss[q]=loss[q]-p1*x[q][t]
    return loss

print(loss(weight,x_data,y_data))

for c in range(10000):
    los = loss(weight,x_data,y_data)
    weight=weight+los*lr

print(los)
temp = weight.dot(x_data)
t=sigmoid(temp)
print(t)
