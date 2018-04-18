function [myNet] = EmptyNetwork(LayerVector)
%Network.m
 % Define an x-layer Network as a structure array
% INPUT: LayerVector - a vector, such as [70,15,10,1], with the number of
%         nodes per layer of the network.  The preceding example would
%         have 4 layers, the first layer (or input layer) with 70 nodes,
%         and so on.
% OUTPUT: Structure array with weights and biases representing
%           the network. Everything is set to zero (hence being empty)
% Created: 2018/04/18, 24 Cummington, Boston
%  Byron Price
% Updated: 2018/04/18
%  By: Byron Price

field = 'Weights';
field2 = 'Biases';
field3 = 'numCalcs';
field4 = 'numLayers';
field5 = 'layerStructure';
field6 = 'numHidden';

numLayers = length(LayerVector);
value = cell(1,1);
value2 = cell(1,1);

value{1} = cell(1,numLayers-1);
value2{1} = cell(1,numLayers-1);

for ii=1:(numLayers-1)
    value{1}{ii} = zeros([LayerVector(ii),LayerVector(ii+1)]);
    value2{1}{ii} = zeros([LayerVector(ii+1),1]);
end
value3 = numLayers-1;
value4 = numLayers;
value5 = LayerVector;
value6 = numLayers-2;
myNet = struct(field,value,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6);
end

