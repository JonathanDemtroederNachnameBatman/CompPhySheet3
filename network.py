import numpy as np
import matplotlib.pyplot as plt

def uniform_weights(size, low=-0.1, high=0.1):
    return np.random.uniform(low=low, high=high, size=size)

def gaus_weights(size, low=-0.1, high=0.1):
    return np.random.normal(loc=0, scale=high/np.sqrt(2*np.pi), size=size) + low

class Network:

    def __init__(self,
                 layer_sizes,
                 activation_function_id,
                 cost_function_id,
                 batch_size,
                 eta,
                 target_function,
                 weight_func=uniform_weights):

        self.layer_sizes = layer_sizes
        self.activation_function = activation_function_id
        self.cost_function_id = cost_function_id
        self.batch_size = batch_size
        self.num_layers = len(layer_sizes) - 1
        self.eta = eta
        self.target_func = target_function

        # init weights and biases

        # Weights is a list of weight matrices (one per transition between layers)
        # Single weights are picked from a uniform distribution
        self.Weights = [weight_func([self.layer_sizes[j], self.layer_sizes[j + 1]]) for j in range(self.num_layers)]

        # Biases is a list of bias vectors (one per layer)
        # Single biases are initialized by value 0
        self.Biases = [np.zeros(self.layer_sizes[j + 1]) for j in range(self.num_layers)]


    def cost_function(self, y_target, y_out_result):

        if self.cost_function_id == 'square':
            return ((y_target - y_out_result) ** 2).sum() / self.batch_size

        elif self.cost_function_id == 'cross_entropy':
            return -(y_target * np.log(np.abs(y_out_result)) + (1 - y_target) * np.log(np.abs(1 - y_out_result))).sum()

        else:
            raise Exception("Enter valid cost-function")


    def non_linear_activation(self, z):

        if self.activation_function == 'sigmoid':
            val = 1 / (1 + np.exp(-z))
            return val, np.exp(-z) * (val ** 2)  # return f, f'

        elif self.activation_function == 'relu':
            return (z > 0)*z, z > 0  # return f, f'

        elif self.activation_function == 'tanh':
            return np.tanh(z), 1 - np.tanh(z)**2

        else:
            raise Exception("Enter valid activation-function")


    def linear_activation(self, y, w, b):
        return np.dot(y, w) + b


    def activation(self, y, w, b):
        return self.non_linear_activation(self.linear_activation(y, w, b))


    def apply_net(self, x):  # one forward pass through the network
        y_layer = []  # to save the neuron values
        df_layer = []  # to save the f'(z) values
        y = x  # start with input values
        y_layer.append(y)
        for w, b in zip(self.Weights, self.Biases):  # loop through all layers
            # j=0 corresponds to the first layer above the input
            y, df = self.activation(y, w, b)  # one step
            df_layer.append(df)  # store f'(z) [needed later in backprop]
            y_layer.append(y)  # store f(z) [also needed in backprop]
        return y, y_layer, df_layer


    def backward_step(self, delta, w, df):
        # delta (batchsize,layersize(N)), w (layersize(N-1),layersize(N))
        # df = df/dz at layer N-1, shape (batchsize,layersize(N-1))
        return np.dot(delta, np.transpose(w)) * df


    def backprop(self, y_target, y_layer, df_layer):
        # one backward pass through the network

        batchsize = np.shape(y_target)[0]
        num_layers = len(self.Weights)  # number of layers excluding input
        dw_layer = [None] * num_layers  # dCost/dw
        db_layer = [None] * num_layers  # dCost/db

        delta = (y_layer[-1] - y_target) * df_layer[-1]
        dw_layer[-1] = np.dot(np.transpose(y_layer[-2]), delta) / batchsize
        db_layer[-1] = delta.sum(0) / batchsize

        for j in range(num_layers - 1):
            delta = self.backward_step(delta, self.Weights[-1 - j], df_layer[-2 - j])
            dw_layer[-2 - j] = np.dot(np.transpose(y_layer[-3 - j]), delta)
            db_layer[-2 - j] = delta.sum(0) / batchsize

        return dw_layer, db_layer  # gradients for weights & biases


    def gradient_step(self, eta, dw_layer, db_layer):
        # update weights & biases (after backprop!)
        num_layers = len(self.Weights)
        for j in range(num_layers):
            self.Weights[j] -= eta * dw_layer[j]
            self.Biases[j] -= eta * db_layer[j]


    def train_batch(self, x, y_target, eta):  # one full training batch
        # x is an array of size batchsize x (input-layer-size)
        # y_target is an array of size batchsize x (output-layer-size)
        # eta is the stepsize for the gradient descent

        y_out_result, y_layer, df_layer = self.apply_net(x)
        dw_layer, db_layer = self.backprop(y_target, y_layer, df_layer)
        self.gradient_step(eta, dw_layer, db_layer)
        cost = self.cost_function(y_target, y_out_result)
        return y_out_result, cost


    def make_batch(self):
        inputs = np.random.uniform(low=-.5, high=+.5, size=[self.batch_size, 2])
        targets = np.zeros([self.batch_size, 1])  # must have right dimensions
        targets[:, 0] = self.target_func(inputs[:, 0], inputs[:, 1])
        return inputs, targets


    def remove_last_hidden_layer(self):
        m = len(self.Weights)
        if m >= 2:

            if m >= 3:
                to_be_removed = self.Weights[m-2]
                to_be_reconnected = self.Weights[m-3]

                # Check if number of neurons is equal
                if np.shape(to_be_reconnected)[0] == np.shape(to_be_removed)[0]:
                    # layer can be removed safely

                    self.Weights.pop(m-2)
                    self.Biases.pop(m-2)
                    self.layer_sizes.pop(m-2)
                    self.num_layers -= 1
                    print('layer removed')

                if np.shape(to_be_removed)[0] > np.shape(to_be_reconnected)[0]:
                    # drop additional weights first

                    neurons_to_be_removed = np.shape(to_be_removed)[0] - np.shape(to_be_reconnected)[0]
                    for i in range(neurons_to_be_removed):
                        np.delete(self.Weights[m - 1], np.shape(self.Weights[m - 1])[0] - 1)
                        np.delete(self.Biases[m - 1], np.shape(self.Biases[m - 1])[0] - 1)

                    self.Weights.pop(m-2)
                    self.Biases.pop(m-2)
                    self.layer_sizes.pop(m-2)
                    self.num_layers -= 1
                    print('layer removed')

                elif np.shape(to_be_removed)[0] < np.shape(to_be_reconnected)[0]:
                    # missing information

                    raise Exception("Unequal shapes: Cant remove layer")

            else:
                self.Weights.pop(m - 2)
                self.Biases.pop(m - 2)
                self.layer_sizes.pop(m - 2)
                self.num_layers -= 1
                print('layer removed')


    def train_network(self, batches, plot=True):
        costs = []
        for k in range(batches):
            x, y_target = self.make_batch()
            y_out_result, cost = self.train_batch(x, y_target, self.eta)
            costs.append(cost)
        if plot:
            plt.plot(costs)
            plt.title("Cost function during training")
            plt.show()
        return costs
