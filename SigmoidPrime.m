function [Y] = SigmoidPrime(Z)
%SigmoidPrime.m
%   Derivative of the sigmoid function.
Y = (exp(-Z))./((1+exp(-Z)).*(1+exp(-Z)));
end

