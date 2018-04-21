function [y] = SwishPrime(Z)
%SwishPrime.m
% Calculate the derivative of the Swish function for an input
%  fix the tunable parameter beta to be 1
fx = Swish(Z);
y = fx+(1+exp(-Z)).^(-1).*(1-fx);
end


