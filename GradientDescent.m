function [Network] = GradientDescent(Network,dCostdWeight,dCostdBias,m,eta)
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
%
%OUTPUT: Network - the same as the input 'Network' but with modified
%          weights and biases
%
% Created: 2016/02/05, 24 Cummington, Boston
%  Byron Price
% Updated: 2016/02/08, 24 Cummington, Boston
% By: Byron Price

numCalcs = size(Network.Weights,2);

for ii=1:numCalcs
    w = (Network.Weights{ii});
    b = Network.Biases{ii};
    Network.Weights{ii} = w - (eta/m).*dCostdWeight{ii};
    Network.Biases{ii} = b - (eta/m).*dCostdBias{ii};
end
end

