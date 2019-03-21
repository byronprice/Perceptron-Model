function [dropOutNet,indices] = MakeDropOutNet(Network,alpha)
%MakeDropOutNet.m
 % Create a smaller network using the original network to perform drop-out
 %  during learning (randomly omit hidden nodes) ... network must have at
 %  least 3 layers
% INPUT: Network - the original network structure (output from Network.m)
%        alpha - the proportion of hidden units to keep (must be between 0
%        and 1)
% OUTPUT: dropOutNet - Network structure array that is smaller than the original
%         indices - indices from the original network, so we can insert
%         this smaller network back into the original one after an
%         iteration through the mini-batches during SGD
% Created: 2018/04/18, 24 Cummington, Boston
%  Byron Price
% Updated: 2018/04/18
%  By: Byron Price

% if Network.numHidden==0
%    fprintf('Error: Network must have at least 1 hidden layer\n');
%    return;
% end
newStruct = Network.layerStructure;
for ii=1:Network.numHidden-1
   newStruct(ii+1) = round(Network.layerStructure(ii+1)*alpha);
end

dropOutNet = EmptyNetwork(newStruct);

indices = cell(Network.numLayers,1);
for ii=1:Network.numLayers
    if ii==1 || ii>=(Network.numLayers-1)
        indices{ii} = 1:Network.layerStructure(ii);
    else
         indices{ii} = randperm(Network.layerStructure(ii),dropOutNet.layerStructure(ii));
    end
end

for ii=1:Network.numCalcs
   dropOutNet.Weights{ii} = Network.Weights{ii}(indices{ii},indices{ii+1});
   dropOutNet.Biases{ii} = Network.Biases{ii}(indices{ii+1});
end

end

