'''
NOTES - Working presets:
    - Dataset size 100 (all training data)
    - 3 layer, 2 sigmoid activations, and tanh for final activation
    - epochs = 1000
    - learning rate = 0.1
    - Takes roughly 10 seconds to train the network
    - 10 ms per epoch roughly
'''

import NNModular as mnn
import NNDisplay as dnn
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

# Solving sine - this is my own attempt to use the resources given to predict sine

# 3 functions - create dataset, create neural network, train network
# 1 display class for both the loss over time and the actual function

pi = math.pi

# Create dataset:
def create_dataset():
    dataset = []
    answers = []
    
    for x in range(0, 100):
        dataset.append([pi * 2 * random.random()])
    for x in range(0, 100):
        answers.append(np.sin(dataset[x]))

    dataset = np.reshape(dataset, (100, 1, 1))
    answers = np.reshape(answers, (100, 1, 1))

    data = list(zip(dataset, answers))
    train_data = []
    test_data = []

    for x in range(0, 100):
        if x <= 66:
            train_data.append(data[x])
        if x > 66:
            test_data.append(data[x])
        
    return [test_data, train_data]    

def create_network():
    # Making network framework
    network = [
        mnn.Dense(1, 8),
        mnn.Sigmoid(),
        mnn.Dense(8, 8),
        mnn.Sigmoid(),
        mnn.Dense(8, 1),
        mnn.Tanh()
    ]
    return network
    
def train_network(neural_network, dataset):
    # Training the neural network
    
    # Setting control variables
    epochs = 10000
    learning_rate = 0.1
    
    for e in range(epochs):
        #print("_____________________Epoch: ", e+1,"_______________________")
        error = 0
        for x, y in dataset:
            output = x
            for layer in network:
                output = layer.forward(output)
    
            # Calculating error
            error = mnn.mse(y, output)

            grad = mnn.mse_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

            error /= len(x)
            #print('error=%f' % (error), 'x=',x,' ,ypred=', output)
        #dnn.Display.function_graph(dataset, network)

dataset = create_dataset()
network = create_network()

print("Loading...")
start_time = time.perf_counter()
train_network(network, dataset[1])
end_time = time.perf_counter()
print("Time: ", end_time - start_time)

dnn.Display.function_graph(dataset[0], network)
dnn.Display.display_parameters(network)




