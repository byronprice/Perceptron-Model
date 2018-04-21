function [Output,Z] = Feedforward(Input,Network)
%Feedforward.m
% This code will take as input a network model and
%   also a series of inputs to its first layer.  It
%   will then calculate a feedforward output from the network assuming
%   sigmoid neurons at each of the network's nodes.
%INPUT: Input - vector of inputs to the network, which 
%        in this case will be pixel values from a 28x28 
%        matrix converted to vector form
%         if matrix = size(28,28);
%         vector = reshape(matrix,[28*28,1]);
%       Network - structure array representing our
%          Network, see the function Network
%
%OUTPUT: Output - cell array of the outputs from each layer of the network,
%           starting with the second layer (the first layer is purely an
%           input layer).
%        Z - cell array of the weighted and biased inputs to each layer,
%           starting with the second layer.
% 
% Created: 2016/02/05, 24 Cummington, Boston
%  Byron Price
% Updated: 2016/02/08
%  By: Byron Price

Output = cell(1,Network.numCalcs);
Z = cell(1,Network.numCalcs);

X = Input;
for ii=1:Network.numCalcs
    Z{ii} = ((Network.Weights{ii})')*X+Network.Biases{ii};
    Output{ii} = SoftPlus(Z{ii});
    X = Output{ii};
end
end

