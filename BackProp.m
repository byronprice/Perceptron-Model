function [dCostdWeight,dCostdBias] = BackProp(Input,Network,DesireOutput)
%BackProp.m
%  See http://neuralnetworksanddeeplearning.com/chap2.html for more
%   information on the back propagation algorithm.
%INPUT: Input - input vector to the neural network
%       Network - structure array representing the network
%          see the functions Network and Feedforward
%       DesireOutput - vector representing the correct output given the
%          input "Input" ... this is for the training/learning phase
%
%OUTPUT: dCostdWeight - cell array with matrices representing the partial 
%          derivative of the cost function with respect to the weights
%        dCostdBias - same as dCostdWeight, but for the biases
%
% Created: 2016/02/05, 24 Cummington, Boston
%  Byron Price
% Updated: 2016/10/13
%  By: Byron Price

[Output,Z] = Feedforward(Input,Network);

Activations = cell(1,Network.numCalcs);
Activations{1} = Input;
for ii=2:Network.numCalcs
    Activations{ii} = Output{ii-1};
end

dCostdWeight = cell(1,Network.numCalcs);
dCostdBias = cell(1,Network.numCalcs);

% tmp = Output{end}; %apply softmax
% softmaxout = exp(tmp)./sum(exp(tmp));

[deltaL] = MeanSquareT2(Output,DesireOutput,Z);

dCostdWeight{end} = Activations{end}*deltaL';
dCostdBias{end} = deltaL;

for ii=Network.numCalcs:-1:2
    deltaL = (Network.Weights{ii}*deltaL).*SwishPrime(Z{ii-1});
    
    dCostdWeight{ii-1} = Activations{ii-1}*deltaL';
    dCostdBias{ii-1} = deltaL;
end
end

function [deltaL] = CrossEntropy(Output,DesireOutput,Z)
deltaL = -DesireOutput./Output{end}+(1-DesireOutput)./(1-Output{end});
deltaL = deltaL.*SwishPrime(Z{end});
end

function [deltaL] = MeanSquare(Output,DesireOutput,Z)
deltaL = (Output{end}-DesireOutput).*SwishPrime(Z{end});
end

function [deltaL] = MeanSquareT2(Output,DesireOutput,Z)
deltaL = (Output{end}-DesireOutput);
end

function [deltaL] = MeanSquareLinearOut(Output,DesireOutput,Z)
deltaL = (Z{end}-DesireOutput); % linear output neuron, mean-squared error cost
end

function [deltaL] = PoissonDev(Output,DesireOutput,Z)
deltaL = exp(Z{end})-DesireOutput; % Poisson deviance cost function
                                           % with exponential output neurons
end

function [deltaL] = LogisticLoss(Output,DesireOutput,Z)

deltaL = (Sigmoid(Z{end})-DesireOutput);

end

% cross-entropy cost function
%  with neuron function a and desired output y
%  a might be the sigmoid function for example
% C = -1/n * SUM_x [yln(a)+(1-y)ln(1-a)]
%  in the output layer, we have
% dCostdWeight = 1/n * SUM_x [a(L-1)*(a(L)-y)]
% dCostdBias = 1/n * SUM_x [a(L)-y]
%  this corresponds to the exact same as above for the
%  quadratic cost function but with no multiplication
%  by sigmoid prime

%softmax cost function 
% outputs are exponentials that sum to 1
% cost is -ln(activation at desired output)
% dCostdWeight and dCostdBias as above

