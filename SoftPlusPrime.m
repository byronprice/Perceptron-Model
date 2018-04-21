function [y] = SoftPlusPrime(Z)
%SoftPlus.m
% Calculate the derivative of the SoftPlus function for an input.

y = 1./(1+exp(-Z));
end
