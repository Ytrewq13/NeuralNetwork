import numpy as np
import matplotlib.pyplot as plt

# Set random seed.
np.random.seed(1)

# Structure of the network.
layers = [2, 4, 1] # in, hidden, out
# Sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# Input dataset.
X = np.array([ [0,1],[1,0] ]) # inputs for NOT data.
X = np.array([ [0,0],[0,1],[1,0],[1,1] ]) # inputs for XOR data.
# output dataset.
y = np.array([ [1,0],[0,1] ]) # outputs for NOT data.
y = np.array([ [0],[1],[1],[0] ]) # outputs for XOR data.

# Init synapse weights.
syn0 = 2*np.random.random((layers[0],layers[1])) - 1 # First layer.
syn1 = 2*np.random.random((layers[1],layers[2])) - 1 # Second layer.

# Scalable version.
syn = []
for l in range(len(layers)-1):
    print(l)
    syn.append(2*np.random.random((layers[l], layers[l+1])) - 1)

errors = []

learn_rate = 0.01

plt.ion()

for i in xrange(2000001):
    # Scalable version.
    l = [X]
    for s in syn:
        l.append(nonlin(np.dot(l[-1],s)))

    final_error = y - l[-1] # Scalable version.

    if (i % 10000 == 0):
        error = np.mean(np.abs(final_error))
        print("Error2:" + str(error)) # test.
        errors.append(error)
        plt.axis([0,i,0,0.1])
        plt.scatter(i,error)
        plt.draw()
        plt.pause(0.05)


    # Scalable version
    final_delta = final_error*nonlin(l[-1],deriv=True)

    errors = [final_error] # Scalable.
    deltas = [final_delta]
    for j in reversed(range(1,len(syn))):
        error = deltas[-1].dot(syn[j].T) # The error.
        delta = error * nonlin(l[j],deriv=True)
        errors.append(error)
        syn[j] += learn_rate * l[j].T.dot(deltas[-1])
        deltas.append(delta)
    syn[0] += learn_rate * np.dot(l[0].T,deltas[-1])


while True:
    plt.pause(0.05)






















# Whitespace.
