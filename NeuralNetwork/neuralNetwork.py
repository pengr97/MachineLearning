import numpy as np
import scipy.special as spFunc
import matplotlib.pyplot as plt

class neuralNetwork:

    # initialize the model
    def __init__(self,inputNodes,hiddenNodes,outputNodes,learningRate):

        self.inNodes = inputNodes       # number of input node
        self.hnNodes = hiddenNodes      # number of hidden node
        self.onNodes = outputNodes      # number of output node

        # w_ih: the weight matrix between input node and hidden node
        # w_ho: the weight matrix between hidden node and output node
        self.w_ih = np.random.normal(0.0,pow(self.hnNodes,-0.5),(self.hnNodes,self.inNodes))     # initialize the w_ih by normal distribution
        self.w_ho = np.random.normal(0.0,pow(self.onNodes,-0.5),(self.onNodes,self.hnNodes))     # initialize the w_ho by normal distribution

        # lr: the learning rate of the neuralNetwork
        self.lr = learningRate

        # activation_func: the activation function
        self.activation_func = lambda x: spFunc.expit(x)

        pass

    # model train
    def train(self,input_list,target_list):

        # transform input and target list to numpy.array
        inputs = np.array(input_list,ndmin=2).T
        targets = np.array(target_list,ndmin=2).T

        # calculate the input and output of hidden nodes
        hidden_inputs = np.dot(self.w_ih,inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        # calculate the input and output of final output nodes
        final_inputs = np.dot(self.w_ho,hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        # calculate the error between targets and final_outputs
        output_errors = targets-final_outputs

        # calculate the error between hidden node and input node
        hidden_errors = np.dot(self.w_ho.T,output_errors)

        self.w_ho += self.lr*np.dot((output_errors*final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))

        self.w_ih += self.lr*np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs))

        pass

    # model predict
    def query(self,input_list):

        # transform input list to numpy.array
        inputs = np.array(input_list,ndmin=2).T

        # calculate the input of hidden nodes
        hidden_input = np.dot(self.w_ih,inputs)
        # calculate the output of hidden nodes after the activation function
        hidden_output = self.activation_func(hidden_input)

        # calculate the input of the final output nodes
        final_input = np.dot(self.w_ho,hidden_output)
        # calculate the output of the final output nodes after the activation function
        final_output = self.activation_func(final_input)

        return final_output

if __name__ == "__main__":

    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    # initialize the model
    model = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate);

    # train the model
    for train_data in open("mnist_dataset/mnist_train.csv"):
        number = train_data.split(',')
        inputs = (np.asfarray(number[1:])/255.0 * 0.99)+0.01
        targets = np.zeros(output_nodes)+0.01
        targets[int(number[0])]=0.99

        model.train(inputs,targets)

    # test the model
    for test_data in open("mnist_dataset/mnist_test.csv"):
        number = test_data.split(',')
        inputs = (np.asfarray(number[1:])/255.0 * 0.99)+0.01

        outputs = model.query(inputs)

        # print(outputs)
        print(np.argmax(outputs.T[0]))      # print the test result

    # draw the real result
    for test_data in open("mnist_dataset/mnist_test.csv"):
        number = test_data.split(',')
        img_array = np.asfarray(number[1:]).reshape((28,28))
        plt.imshow(img_array,cmap="Greys",interpolation="None")
        plt.show()