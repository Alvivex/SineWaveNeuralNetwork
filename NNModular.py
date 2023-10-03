# Neural network from sratch - "The Independent Code" YouTube tutorial
'''

NOTES FROM VIDEO:

Steps:
1 - Feed input, data flowas from layer to layer, retrieve output
2 - Calculate the error (e.g. 1/2(y*-y)^2)
3 - Adjust parameters using gradient descent (w <- w -a(de/dw)) (alpha is control)
4 - Start again
MOST IMPORTANT PART IS STEP 3


For modular neural network design:
- Need to implement each layer separately
- For each layer:
    - Whenever input is given, an output is given (forward propagation)
        x --> Layer --> y
    de/dx <-- Layer <-- de/dy
    - For each parameter in the layer:
        - The layer must compute the derivative of the error with respect to parameter (e.g. weight or bias)
        - de/dw = (de/dy)(dy/dw)
        - de/dx = (de/dy)(dy/dx) (The input for the next layer on left)
        - The de/dx allows for all the layers to update parameters


DERIVATIVES:

From (de/dy), need to calculate de/dw and de/db, as well as de/dx for next layer

Weights:
de/dw is a matrix with the same shape as the weights column
de/dwji = (de/dyji)xi
de/dw = (de/dy)X^T (X^T is the input vector X transposed)

Biases:
de/db = (de/dy)(de/db) = (de/dy)(1) = de/dy

Next Layer:
de/dx = (W^T)(de/dy) (W^T is the weight matrix for that layer transposed)

FOR ACTIVATION FUNCTIONS:

Tanh:
de/dx = (de/dy)(f'(X))

Mean Squared Error (MSE):
- This is the final equation used, so it cannot use any next layer derivative

de/dy = (2/n)(y-y*) for n being the size of the output

'''

# CODE:

import numpy as np

# The base layer (has children dense and activation)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input):
        # return output
        pass
    def backward(self, output_gradient, learning_rate):
        # update parameters and return input gradient
        pass

# The dense layer
# The dense layer connects i input neurons to x output neurons
# The connections between each neuron is reprented by a numerical weight
# wji is a weight connecting layer i to layer j

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        # See below for derivative calculations
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        
        # The return returns the de/dy for the next layer
        return np.dot(self.weights.T, output_gradient)

# The activation functions are kept separate from the dense layer
# The activation layer is a separate layer
# Activation layer takes input neurons and passes through activation function
# Output has same shape as input

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        # The activation_prime is the derivative of the function

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: (1 / (np.cosh(x) **2))
        super().__init__(tanh, tanh_prime)

class SlackSigmoid(Activation):
    def __init__(self):
        tanh = lambda x: (2/(1 + np.exp(-x))) - 1
        tanh_prime = lambda x: 2 * (np.exp(-x)) * ((1 + np.exp(-x))**-2)
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        sigmoid_prime = lambda x:  (1/(1 + np.exp(-x))) * (1 - (1/(1 + np.exp(-x))))
        super().__init__(sigmoid, sigmoid_prime)

# As mse is not calculated like an activation layer, it is separate:

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size (y_true)
        
        
