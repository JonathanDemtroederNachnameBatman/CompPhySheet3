
import numpy as np
#from numba import jit

#@jit()
def relu(z): # calculate f(z) and f'(z)
    return (z>0)*z,z>0

#@jit()
def sigmoid(z):
    return 1/(1+np.exp(-z))

#@jit()
def xor(x1, x2):
    a = x1 * x2
    b1 = a < 0
    b2 = a >= 0
    a[b1] = 1
    a[b2] = 0
    return a

def gen_data():
    x1 = np.random.uniform(-10, 10, 50)
    x2 = np.random.uniform(-10, 10, 50)
    x1, x2 = np.meshgrid(x1, x2)
    y = xor(x1, x2)
    pos = np.vstack([x1.ravel(), x2.ravel()]).T
    y_flat = y.flatten()
    np.savetxt('data/in.txt', pos)
    np.savetxt('data/out.txt', y_flat)

def forward_step(y,w,b, activation=relu):
    z = np.dot(w, y) + b
    return activation(z) # apply nonlinearity and return result

def apply_net(x, Weights, Biases):  # one forward pass through the network
    y_layer = []  # to save the neuron values
    df_layer = []  # to save the f'(z) values
    y = x  # start with input values
    y_layer.append(y)
    for w, b in zip(Weights, Biases):  # loop through all layers
        # j=0 corresponds to the first layer above the input
        y, df = forward_step(y, w, b)  # one step
        df_layer.append(df)  # store f'(z) [needed later in backprop]
        y_layer.append(y)  # store f(z) [also needed in backprop]
    return y, y_layer, df_layer

def apply_net_simple(x, Weights, Biases):  # one forward pass through the network
    # no storage for backprop (this is used for simple tests)

    y = x  # start with input values
    for w, b in zip(Weights, Biases):  # loop through all layers
        # j=0 corresponds to the first layer above the input
        y, df = forward_step(y, w, b)  # one step
    return y

def backward_step(delta, w, df):
    # delta (batchsize,layersize(N)), w (layersize(N-1),layersize(N))
    # df = df/dz at layer N-1, shape (batchsize,layersize(N-1))
    return np.dot(delta, np.transpose(w)) * df

def backprop(y_target, y_layer, df_layer, Weights, Biases):
    # one backward pass through the network

    batchsize = np.shape(y_target)[0]
    num_layers = len(Weights)  # number of layers excluding input
    dw_layer = [None] * num_layers  # dCost/dw
    db_layer = [None] * num_layers  # dCost/db

    delta = (y_layer[-1] - y_target) * df_layer[-1]
    dw_layer[-1] = np.dot(np.transpose(y_layer[-2]), delta) / batchsize
    db_layer[-1] = delta.sum(0) / batchsize

    for j in range(num_layers - 1):
        delta = backward_step(delta, Weights[-1 - j], df_layer[-2 - j])
        dw_layer[-2 - j] = np.dot(np.transpose(y_layer[-3 - j]), delta)
        db_layer[-2 - j] = delta.sum(0) / batchsize

    return dw_layer, db_layer  # gradients for weights & biases

def gradient_step(eta, dw_layer, db_layer, Weights, Biases):
    # update weights & biases (after backprop!)
    num_layers = len(Weights)
    for j in range(num_layers):
        Weights[j] -= eta * dw_layer[j]
        Biases[j] -= eta * db_layer[j]
    return Weights, Biases

def train_batch(x, y_target, eta, Weights, Biases, batchsize):  # one full training batch
    # x is an array of size batchsize x (input-layer-size)
    # y_target is an array of size batchsize x (output-layer-size)
    # eta is the stepsize for the gradient descent

    y_out_result, y_layer, df_layer = apply_net(x, Weights, Biases)
    dw_layer, db_layer = backprop(y_target, y_layer, df_layer, Weights, Biases)
    Weights, Biases = gradient_step(eta, dw_layer, db_layer, Weights, Biases)
    cost = ((y_target - y_out_result) ** 2).sum() / batchsize
    return y_out_result, cost, Weights, Biases
