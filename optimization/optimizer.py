#Plot3D[(x + 2*y - 7)^2 + (2*x + y - 5)^2, {x, -5, 5}, {y, -5, 5}]
import numpy
def fx(x,y):
    return 4 * (-5 + 2 * x + y) + 2 * (-7 + x + 2 * y)

def fy(x,y):
    return 2 * (-5 + 2 * x + y) + 4 * (-7 + x + 2 * y)

'''[ fxx,fxy
    fyx,fyy ]'''
hessian = numpy.array([[10,8],[8,10]])
#sgd
lr = 0.1
s = numpy.array([4,1])
end = numpy.array([1,3])
v=0

for i in range(0,10000):
    dx=fx(s[0],s[1])
    dy=fy(s[0],s[1])
    step = lr * numpy.array([dx,dy])
    s=s-step
    if numpy.sum((s-end)*(s-end))<0.00001:
        print(i)
        break


#momentum
lr = 0.1
s = numpy.array([4,1])
v=0
for i in range(0,10000):
    dx=fx(s[0],s[1])
    dy=fy(s[0],s[1])
    step = lr * numpy.array([dx,dy])
    v=0.3*v-step
    s=s+v
    if numpy.sum((s-end)*(s-end))<0.00001:
        print(i)
        break

#Adagrad
lr = 1
s = numpy.array([4,1])
epsilon=10**-7
r=numpy.array([0,0])
for i in range(0,10000):
    dx=fx(s[0],s[1])
    dy=fy(s[0],s[1])
    r=r+numpy.array([dx,dy])*numpy.array([dx,dy])
    step = -lr * numpy.array([dx,dy])/(epsilon+numpy.sqrt(r))
    s=s+step
    if numpy.sum((s-end)*(s-end))<0.00001:
        print(i)
        break

#Adam
lr = 10
s = numpy.array([4,1])
epsilon=10**-4
sg=numpy.array([0,0])
rg=numpy.array([0,0])
r1=0.9
r2=0.999
for i in range(1,1000):
    dx=fx(s[0],s[1])
    dy=fy(s[0],s[1])
    g=numpy.array([dx,dy])
    sg=r1*sg+(1-r1)*g
    rg=r2*rg+(1-r2)*(g*g)

    sg1=sg/(1-r1**i)
    rg1=rg/(1-r2**i)


    step=-lr*sg1/(numpy.sqrt(rg1)+epsilon)
    s=s+step
    if numpy.sum((s-end)*(s-end))<0.00001:
        print(i)
        break

#newton
lr = 0.1
s = numpy.array([4,1])
for i in range(0,1):
    dx=fx(s[0],s[1])
    dy=fy(s[0],s[1])
    step = numpy.array([dx,dy]).dot(numpy.linalg.inv(hessian))
    s=s-step
    if numpy.sum((s-end)*(s-end))<0.00001:
        print(i)
        break
