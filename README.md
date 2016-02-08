# Perceptron-Model
This code implements a Perceptron neural network model.  It works using the back-propagation algorithm on "sigmoidal" neurons, 
using a mean-squared error cost function.  The code "Perceptron.m" takes training data from the MNIST database and trains
the network using stochastic gradient descent.  It then runs the altered network "myNet" on the MNIST test data.  That data
must be previously downloaded and placed in the same directory.  Here, the files are saved as "TrainingData.mat" and 
"TestData.mat", with the image and label files saved as matrices, "Images" and "Labels".

Place all of the files in the same directory as the training and test data, and run "Perceptron" .

Credit: www.neuralnetworksanddeeplearning.com for an explanation of the back-propagation and stochastic gradient descent
algorithms.

With a few modifications, this code can be made to work for any arbitrarily-sized network (limited, of course, by the computing power available to the train the network).
