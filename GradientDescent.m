function [Network] = GradientDescent(Network,dCostdWeight,dCostdBias,m,eta,n)
%GradientDescent.m
%   GradientDescent algorithm for updating the weights
%    and biases of our network, i.e. learning.
%INPUT: Network - structure array representing the network, see Network
%         function
%       dCostdWeight - output from the function BackProp, these are partial
%         derivatives representing the change in each of the weights of the
%         network
%       dCostdBias - output from the function BackProp, as dCostdWeight but
%         for the biases
%       m - number of training examples per batch, as specified in the 
%         script Perceptron
%       eta - learning rate, again specified in the script Perceptron
%       n - total number of training examples
%
%OUTPUT: Network - the same as the input 'Network' but with modified
%          weights and biases
%
% Created: 2016/02/05, 24 Cummington, Boston
%  Byron Price
% Updated: 2016/10/12
% By: Byron Price

numCalcs = size(Network.Weights,2);
lambda = 10;
for ii=1:numCalcs
    w = (Network.Weights{ii});
    b = Network.Biases{ii};
    Network.Weights{ii} = (1-eta*lambda/n).*w - (eta/m).*dCostdWeight{ii};
    Network.Biases{ii} = b - (eta/m).*dCostdBias{ii};
end
end

%for L2 regularization
%  do (1-eta*lambda/n)*w - ...
%  n is the total number of training examples
%  lambda is called the regularization parameter
