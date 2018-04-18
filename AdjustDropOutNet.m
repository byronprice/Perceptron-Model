function [Network] = AdjustDropOutNet(Network,alpha)
%AdjustDropOutNet.m
 % Divide the weights outgoing from the hidden nodes to compensate for the
 % drop-out procedure
% INPUT: Network - network structure (output by Network.m) after training
%        alpha - proportion of hidden nodes kept during drop-out
%        (between 0 and 1)
% OUTPUT: Network - the original network structure, with updated weights
%          altered to compensate for drop out
% Created: 2018/04/18, 24 Cummington, Boston
%  Byron Price
% Updated: 2018/04/18
%  By: Byron Price

for ii=2:Network.numCalcs
   Network.Weights{ii} = Network.Weights{ii}.*alpha;
end

end