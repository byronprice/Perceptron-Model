function [myNet] = Network(LayerVector)
%Network.m
 % Define an x-layer Network as a structure array
% INPUT: LayerVector - a vector, such as [70,15,10,1], with the number of
%         nodes per layer of the network.  The preceding example would
%         have 4 layers, the first layer (or input layer) with 70 nodes,
%         and so on.
% OUTPUT: Structure array with randomized weights and biases representing
%           the network.  Use standard normal random variables for initial
%           values.
% Created: 2016/02/05, 24 Cummington, Boston
%  Byron Price
% Updated: 2016/02/08
%  By: Byron Price

field = 'Weights';
field2 = 'Biases';

numLayers = length(LayerVector);
value = cell(1,1);
value2 = cell(1,1);

value{1} = cell(1,numLayers-1);
value2{1} = cell(1,numLayers-1);

for ii=1:(numLayers-1)
    value{1}{ii} = normrnd(0,1,[LayerVector(ii),LayerVector(ii+1)]);
    value2{1}{ii} = normrnd(0,1,[LayerVector(ii+1),1]);
end

myNet = struct(field,value,field2,value2);
end

