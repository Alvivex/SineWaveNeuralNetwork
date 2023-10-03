'''
IMPROVEMENTS
- Must decrease sample size and not give whole data set - randomize the X set
- Feed through in specific sample size (4 points, 7 points, 36 points, etc...)

- Traning set 2/3, Testing set 1/3
- Train on all items on the training set
- Test but don't train on items in testing set

'''

import ModularNeuralNetwork as mnn
import DisplayNeuralNetwork as dnn
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Solving sine - this is my own attempt to use the resources given to predict sine

# 3 functions - create dataset, create neural network, train network
# 1 display class for both the loss over time and the actual function

def create_dataset():
    # making the dataset
    xValues = []
    trueyValues = []
    interval = 90
    
    for x in range(0, 361, int(interval)):
        trueyValues.append([math.sin(np.deg2rad(x))])
        xValues.append([x])

    xValues = np.reshape(xValues, (int(360/interval)+1, 1, 1))
    trueyValues = np.reshape(trueyValues, (int(360/interval)+1, 1, 1))

    #Link and randomise the dataset
    dataset = list(zip(xValues, trueyValues))
    #dataset_rand = random.shuffle(dataset)
    
    # Split the dataset into training and testing data
    train_data = []
    test_data = []

    for x in range(0, len(dataset)):
        if x < int((2/3) * len(dataset)):
            train_data.append(dataset[x])
        if x >= int((2/3) * len(dataset)):
            test_data.append(dataset[x])
        
    return [train_data, test_data]

    
def create_network():
    # Making network framework
    network = [
        mnn.Dense(1, 8),
        mnn.Sigmoid(),
        mnn.Dense(8, 8),
        mnn.Sigmoid(),
        mnn.Dense(8, 1),
        mnn.Sigmoid()
    ]
    return network
    
def train_network(neural_network, dataset):
    # Training the neural network
    
    # Setting control variables
    epochs = 100
    learning_rate = 0.9
    
    for e in range(epochs):
        print("_____________________Epoch: ", e+1,"_______________________")
        error = 0
        #xValues_norm = (xValues - np.min(xValues)) / (np.max(xValues) - np.min(xValues))
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


network = create_network()
dataset = create_dataset()

train_network(network, dataset[0])

dnn.Display.function_graph(dataset[0], network)

#train_network(network, dataset)
#dnn.Display.function_graph(dataset, network)
#dnn.Display.display_parameters(network)





