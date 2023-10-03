# 20/9/2023
# This is practice database 1 (Learn to replicate a sine wave with one input)

''' Plan:
Create the framework for the neural network (layers and functions)
Implement Forward Propagation
Implement backward propagation
Create database of inputs and outputs from a calculator, then feed through the network until the network is more accurate
Save the neural network values for each weight and bias so to accurately calculate sine values to an accurate number of sig figs
'''

import numpy as np
import math
import matplotlib.pyplot as plt

# Generate a set of randomly generated floats between 0 and 360, with their correct sin equivalents
class Data_Set:
    def __init__(self, size):
        # Generate inputs
        input_data = []
        for x in range(0, size):
            input_data.append(np.random.uniform(0, 360))
        self.data = input_data
        # Generate correct answers
        correct_answers = []
        for num in input_data:
            correct_answers.append(math.sin(np.deg2rad(num)))
        self.answers = correct_answers

# This class will be used for generating each layer with weights and biases
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, input_neuron_array):
        self.output = np.dot(input_neuron_array, self.weights) + self.biases

# These are the functions used:
# ReLU:
class ReLU:
    def calculate(self, input_neuron_array):
        self.output = np.maximum(0, input_neuron_array)

# Activation function for output neurons:
class OutputFunction:
    def calculate(self, input_neuron):
        # Tanh is used to squeeze the output between -1 and 1
        self.output = np.tanh(input_neuron)

# Loss function:
class SquareOfSubtractedValues:
    def calculate(self, input_neuron, correct_answer):
        return ((correct_answer - input_neuron)** 2)

def init_parameters():
    # Declaring the layers and activations for each layer
    dense1 = Layer_Dense(1, 8)
    activation1 = ReLU()

    dense2 = Layer_Dense(8, 8)
    activation2 = ReLU()

    dense3 = Layer_Dense(8, 1)
    activation3 = OutputFunction()

    loss_function = SquareOfSubtractedValues()
    return dense1, dense2, dense3, activation1, activation2, activation3, loss_function

def forward_prop(X, RX, L1, L2, L3, A1, A2, A3, FL):
    #Running a forward pass
    L1.forward(X) # Calculates the values for the 8 neurons by multiplying X by weights and adding bias
    A1.calculate(L1.output)

    L2.forward(A1.output)
    A2.calculate(L2.output)

    L3.forward(A2.output)
    A3.calculate(L3.output)

    # RX is the actual value of X
    loss = FL.calculate(A3.output, RX)

    #print("Output: ", A3.output)
    #print("Loss: ", loss)
    return A3.output

def display_whole_network(L1, L2, L3):
    
    whole_network = []
    
    for weight_array in L1.weights:
        for weight in weight_array:
            whole_network.append(weight)
    for bias_array in L1.biases:
        for bias in bias_array:
            whole_network.append(bias)
        
    for weight_array in L2.weights:
        for weight in weight_array:
            whole_network.append(weight)
    for bias_array in L2.biases:
        for bias in bias_array:
            whole_network.append(bias)
        
    for weight_array in L3.weights:
        for weight in weight_array:
            whole_network.append(weight)
    for bias_array in L3.biases:
        for bias in bias_array:
            whole_network.append(bias)
        
    for value in whole_network:
        print(value)

    print("Size of whole network array: ", len(whole_network))
'''
def calculate_negative_gradient(L1, L2, L3, A3, correct_answer):
    negative_gradient = []
        for (neuron in L3[1])
'''
# Executing a run
parameters = init_parameters()

dataset1 = Data_Set(3)


'''
for x in range(0, 3):
    input_value = dataset1.data[x]

    print("Run: ", x+1)
    print('Input: ', input_value,', Correct Answer: ', dataset1.answers[x])

    forward_prop(input_value, dataset1.answers[x], parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6])

'''

#display_whole_network(parameters[0], parameters[1], parameters[2])
#calculate_negative_gradient(parameters[0], parameters[1], parameters[2], parameters[5], 6)


# Test the data structure
xValues = []
yValues = []
trueyValues = []

for x in range(0, 360, 4):
    y = forward_prop(x, 0, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6])
    
    trueyValues.append(math.sin(np.deg2rad(x)))
    yValues.append(y[0,0])
    xValues.append(x)
    print("calculated y coord: ", y)

plt.plot(xValues, yValues, marker = 'o', ms = 3)
plt.plot(xValues, trueyValues, marker = 'o', ms = 3)
plt.show()






