function [y] = Sigmoid(Z)
%Sigmoid.m
% Calculate the sigmoid function for an input.

y = 1./(1+exp(-Z));
end

