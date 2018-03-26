# Perceptron-Model
This code implements a Perceptron neural network model.  It works using the back-propagation algorithm on "sigmoidal" neurons, 
using a cross-entropy cost function (very similar to the mean-squared error cost function, and the two can easily be interchanged).  The code "Perceptron.m" takes training data from the MNIST database and trains
the network using stochastic gradient descent.  It then runs the altered network "myNet" on the MNIST test data.  That data
must be previously downloaded and placed in the same directory.  Here, the files are saved as "TrainingData.mat" and 
"TestData.mat", with the image and label files saved as matrices, "Images" and "Labels".

Place all of the files in the same directory as the training and test data, and run "Perceptron" . As is, it should attain ~96% accuracy and requires maybe 2 minutes to train. The algorithm uses L2 regularization, though it can easily be reverted to remove this additional complexity.

Change the variable "runs" in the stochastic gradient descent algorithm for longer training periods.  With more and more training runs, the network approaches ~98% classification accuracy.

With very few modifications, this code can be made to work for any network, on any classification problem (limited, of course, by the computing power available to the train the network and by the difficulty of the problem).

Credit: www.neuralnetworksanddeeplearning.com for a simple explanation of the back-propagation and stochastic gradient descent
algorithms. http://www.deeplearningbook.org/ for theoretical understanding of neural networks.

See Perceptron_Model.ipynb or Perceptron_Model.pdf for a Jupyter notebook implementing the code, with figures incorporated.
