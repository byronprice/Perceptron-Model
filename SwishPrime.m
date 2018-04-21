function [y] = SwishPrime(Z)
%SwishPrime.m
% Calculate the derivative of the Swish function for an input
%  fix the tunable parameter beta to be 1
y = Z.*(1./(1+exp(-Z)))+(1./(1+exp(-Z))).*(1-Z.*(1./(1+exp(-Z))));
end


