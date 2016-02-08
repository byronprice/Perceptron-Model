function [dCostdWeight,dCostdBias] = BackProp(Input,Network,DesireOutput)
%BackProp.m
%  See http://neuralnetworksanddeeplearning.com/chap2.html for more
%   information on the back propagation algorithm.
%INPUT: Input - input vector to the neural network
%       Network - structure array representing the network
%          see the functions Network and Feedforward
%       DesireOutput - vector representing the correct output given the
%          input "Input" ... this is for the training/learning phase
%
%OUTPUT: dCostdWeight - cell array with matrices representing the partial 
%          derivative of the cost function with respect to the weights
%        dCostdBias - same as dCostdWeight, but for the biases
%
% Created: 2016/02/05, 24 Cummington, Boston
%  Byron Price
% Updated: 2016/02/08
%  By: Byron Price

[Output,Z] = Feedforward(Input,Network);
numCalcs = size(Network.Weights,2);

Activations = cell(1,numCalcs);
Activations{1} = Input;
for ii=2:numCalcs
    Activations{ii} = Output{ii-1};
end

dCostdWeight = cell(1,numCalcs);
dCostdBias = cell(1,numCalcs);

deltaL = (Output{end}-DesireOutput).*SigmoidPrime(Z{end});
dCostdWeight{end} = Activations{end}*deltaL';
dCostdBias{end} = deltaL;

for ii=numCalcs:-1:2
    W = Network.Weights{ii};
    deltaL = (W*deltaL).*SigmoidPrime(Z{ii-1});

    dCostdWeight{ii-1} = Activations{ii-1}*deltaL';
    dCostdBias{ii-1} = deltaL;
end
end

