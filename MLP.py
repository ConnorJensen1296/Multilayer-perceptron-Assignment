from numpy import exp, array, random, dot
import numpy as np


class randomise():
    # Creating the neuron layer and randomising the weights.
    def __init__(self, numberNeurons, numberInputs):
        self.weights = 2 * random.random((numberInputs, numberNeurons)) - 1


class NeuralNetwork():
    def __init__(self, lowerLayer, upperLayer):

        self.lowerLayer = lowerLayer
        self.upperLayer = upperLayer

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        return np.exp(x) / np.sum(top)

    # We train the neural network through a process of trial and error.
    # Adjusting the learning rate each time.
    def train(self, training_set_inputs, training_set_outputs, maxEpochs,test):
        error_array = np.zeros([])
        for e in range(maxEpochs):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.predict(training_set_inputs)

            # Calculate the error for the upper layer (The difference between the desired output
            # and the predicted output).
            upperLayer_error = training_set_outputs - output_from_layer_2
            upperLayer_delta = upperLayer_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for lower layer (By looking at the weights in the upper layer,
            # we can determine by how much lower layer contributed to the error in upper layer).
            lowerLayer_error = upperLayer_delta.dot(self.upperLayer.weights.T)
            lowerLayer_delta = lowerLayer_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            lowerLayer_adjustment = training_set_inputs.T.dot(lowerLayer_delta)
            upperLayer_adjustment = output_from_layer_1.T.dot(upperLayer_delta)

            # Adjust the weights.
            self.lowerLayer.weights += lowerLayer_adjustment
            self.upperLayer.weights += upperLayer_adjustment
            error_array = np.append(error_array,upperLayer_error[0][0])

        np.savetxt("epoch_error_"+test+".csv", error_array, delimiter=",")

    def predict(self, inputs):
        output_from_lowerLayer = self.__sigmoid(dot(inputs, self.lowerLayer.weights))
        output_from_upperLayer = self.__sigmoid(dot(output_from_lowerLayer, self.upperLayer.weights))
        return output_from_lowerLayer, output_from_upperLayer

if __name__ == "__main__":
    # Setting a random seed
    np.random.seed(1234)

    # Setting the number of epochs
    maxEpochs = 1000
    # Number of Inputs
    NI = 2
    # Number of Hidden Units
    NH = 3
    # Number of Outputs
    NO = 1

    # Create the lower layer with NH neurons, each with NI inputs
    lowerLayer = randomise(NH, NI)

    # Create the higher layer with NO neurons, each with NH inputs)
    upperLayer = randomise(NO, NH)

    # Combine the lower and higher layers to create the neural network
    neuralNetwork = NeuralNetwork(lowerLayer, upperLayer)

    # The training set.
    training_set_inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T
    neuralNetwork = NeuralNetwork(lowerLayer, upperLayer)
    # Train the neural network using the training set.
    neuralNetwork.train(training_set_inputs, training_set_outputs, maxEpochs,test='test1')
    print("---------------------------------------------------------")
    print("Test 1:")
    print("Input of [X,Y] has an output of [Z]")
    hidden_state, output = neuralNetwork.predict(array([0, 0]))
    print("[0, 0] -> ", output)
    hidden_state, output = neuralNetwork.predict(array([0, 1]))
    print("[0, 1] -> ", output)
    hidden_state, output = neuralNetwork.predict(array([1, 0]))
    print("[1, 0] -> ", output)
    hidden_state, output = neuralNetwork.predict(array([1, 1]))
    print("[1, 1] -> ", output)



    print("-------------------------- \n Test 2")
    np.random.seed(12)
    # Setting the number of epochs
    maxEpochs = 1000
    # Number of Inputs
    NI = 4
    # Number of Hidden Units
    NH = 5
    # Number of Outputs
    NO = 1

    # Create the lower layer with NH neurons, each with NI inputs
    lowerLayer = randomise(NH, NI)

    # Create the higher layer with NO neurons, each with NH inputs)
    upperLayer = randomise(NO, NH)

    # Combine the lower and higher layers to create the neural network
    neuralNetwork = NeuralNetwork(lowerLayer, upperLayer)

    # Creating the random test set
    #low=-1.0, high=1.0, size=(200,4)
    RandomVectors = np.random.rand(200,4)
    def SinVector(RandomVectors):
        output = RandomVectors[0] - RandomVectors[1] + RandomVectors[2] - RandomVectors[3]
        return np.sin(output)

    RandomVectorsOutput = np.apply_along_axis(SinVector, 1, RandomVectors)
    RandomVectorsOutput = np.reshape(RandomVectorsOutput, (200,1))

    # The training set.
    training_set_inputs = RandomVectors[0:150]
    training_set_outputs = RandomVectorsOutput[0:150]
    # The test set.
    test_set_inputs = RandomVectors[-50:]
    test_set_outputs = RandomVectorsOutput[-50:]
    neuralNetwork.train(training_set_inputs, training_set_outputs, maxEpochs,test='test2')
    error = np.zeros((50,2))
    for x in range(1,test_set_outputs.size - 1):
        print("--------")
        output = neuralNetwork.predict(test_set_inputs[x])
        print("The predicted output is {} \n The actual output is {}".format(output[1][0],test_set_outputs[x][0]))
        error[x,0] = output[1][0]
        error[x,1] = test_set_outputs[x][0]
    np.savetxt("PredictvsOutputTest2.csv", error, delimiter=",")

    print("-------------------------- \n Test 3")

    """
    # Setting the number of epochs
    maxEpochs = 2500
    # Number of Inputs
    NI = 16
    # Number of Hidden Units
    NH = 10
    # Number of Outputs
    NO = 26
    # Create the lower layer with NH neurons, each with NI inputs
    lowerLayer = randomise(NH, NI)

    # Create the higher layer with NO neurons, each with NH inputs)
    upperLayer = randomise(NO, NH)

    # Combine the lower and higher layers to create the neural network
    neuralNetwork = NeuralNetwork(lowerLayer, upperLayer)

    # Reading and formatting the data.

    def scale(X, x_min, x_max):
        nom = (X - X.min(axis=0)) * (x_max - x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom == 0] = 1
        return x_min + nom / denom

    data_inputs = np.loadtxt("letter-recognition.data", delimiter=",",
                                     usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
    data_outputs = np.loadtxt("letter-recognition.data", delimiter=",", usecols=0, dtype=np.str)
    data_outputs = np.reshape(data_outputs, (20000, 1))
    output = np.zeros((20000, 1), dtype=np.longdouble)
    for x in range(0, data_outputs.size -1):
        output[x] = ord(data_outputs[x][0])
    # The training set.
    output = scale(output, 0, 1)
    training_set_inputs = data_inputs[0:14999]
    training_set_outputs = output[0:14999]

    # The test set.
    test_set_inputs =data_inputs[-5000:]
    test_set_outputs = output[-5000:]

    neuralNetwork.train(training_set_inputs, training_set_outputs, maxEpochs, test = 'test3')
    for x in range(0,test_set_outputs.size - 1):
        print("--------")
        output = neuralNetwork.predict(test_set_inputs[x])
        print("The predicted output is {} \n The actual output is {}".format(output[1:x],test_set_outputs[x]))
    """