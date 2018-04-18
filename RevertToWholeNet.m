function [Network] = RevertToWholeNet(dropOutNet,Network,indices)
%RevertToWholeNet.m
 % Get back to the original network after performing drop-out
 %  during learning (randomly omit hidden nodes) ... network must have at
 %  least 3 layers  (insert the smaller network back into the bigger one)
% INPUT: dropOutNet - the smaller network structure (output from
%          MakeDropOutNet.m)
%        Network - the original network structure (output from Network.m)
%        indices - indices output from MakeDropOutNet.m
% OUTPUT: Network - the original network structure, with updated weighted
%          and biases taken from the dropOutNet
% Created: 2018/04/18, 24 Cummington, Boston
%  Byron Price
% Updated: 2018/04/18
%  By: Byron Price

for ii=1:Network.numCalcs
   Network.Weights{ii}(indices{ii},indices{ii+1}) = dropOutNet.Weights{ii};
   Network.Biases{ii}(indices{ii+1}) = dropOutNet.Biases{ii};
end

end

