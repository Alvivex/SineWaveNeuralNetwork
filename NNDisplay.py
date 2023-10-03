import matplotlib.pyplot as plt
import NNModular as mnn
import numpy as np

class Display():
    def __init(self):
        pass
    def loss_graph(epoints, losspoints):
        plt.plot(epochSet, lossSet)
        plt.show()

        # Display the outputs in a graph
    def function_graph(dataset, neural_network):
        inputs = []
        answers = []
        predictions = []
        
        for n in range(0, len(dataset)-1):
            inputs.append(dataset[n][0][0,0])

        for n in range(0, len(dataset)-1):
            answers.append(dataset[n][1][0,0])
    
        for n in range(0, len(dataset)-1):
            output = [[float(dataset[n][0])]]

            for layer in neural_network:
                output = layer.forward(output)
            predictions.append(output[0,0])
        
        plt.scatter(inputs, answers, marker = 'o', s = 5, color='red')
        plt.scatter(inputs, predictions, marker = 'o', s = 5, color='blue')
        plt.show()
        plt.figure()
        
    def display_parameters(neural_network):
        layerCount = 1
    
        for layer in neural_network:
            if type(layer) == mnn.Dense:
                print("________________Layer: ", layerCount, "___________________")
                print("Weights: ", layer.weights)
                print("Biases: ", layer.bias)
                layerCount += 1
                
    def answer_from_input(neural_network):
        # Test from input
        for x in range(0, 100):
            xValue = input("Enter a value of x between 0 and 2 pi")

            output = [float(xValue)]

            for layer in neural_network:
                output = layer.forward(output)
    
            print("Prediction: ", output[0, 0])
