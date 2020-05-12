import matplotlib.pyplot as plt
import numpy

# one dimension LR
x = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = 2 * x + 3 + numpy.random.normal(0, 0.5, (10,))
s = x.shape
ancilla = numpy.ones(s)
new_x = numpy.array([ancilla, x])
weight = y.dot(numpy.linalg.pinv(new_x))
plt.plot(x, y)
plt.plot(x, weight[1] * x + weight[0])
plt.show()

# multi dimension
x = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = 2 * x + 3 + x*x + numpy.random.normal(0, 3, (10,))
s = x.shape
ancilla = numpy.ones(s)
new_x = numpy.array([ancilla, x,x*x])
weight = y.dot(numpy.linalg.pinv(new_x))
plt.plot(x, y)
plt.plot(x, weight[2]*x *x + weight[1] * x + weight[0])
plt.show()