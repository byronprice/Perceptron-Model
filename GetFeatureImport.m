function [featImport] = GetFeatureImport(Network,Input,outputDim)
% GetFeatureImport.m
%   feature importance, aka axiomatic feature attribution, from paper
%  Axiomatic Attribution for Deep Networks, Sundararajan et al. 2017
%    works for feedfoward network described in this repository
if nargin<3
    [Output,~] = Feedforward(Input,Network);
    [~,outputDim] = max(Output{end});
end

baselineValue = 0.5; % 0.5 for greyscale baseline, 0 for black
    
numInputDims = Network.layerStructure(1);

numGradCalcs = 5000;
alpha = linspace(0,1,numGradCalcs);

baselineIm = baselineValue.*ones(numInputDims,1);
integral = 0;
for jj=1:numGradCalcs
    [gradient] = FIBackProp(baselineIm+alpha(jj).*(Input-baselineIm),Network,outputDim);
    integral = integral+gradient./numGradCalcs;
end
featImport = (Input-baselineIm).*integral;

end

function [gradient] = FIBackProp(Input,Network,outputDim)
%FIBackProp.m
%  See http://neuralnetworksanddeeplearning.com/chap2.html for more
%   information on the back propagation algorithm.
%INPUT: Input - input vector to the neural network
%       Network - structure array representing the network
%          see the functions Network and Feedforward
%
%OUTPUT: gradient - gradient of the output with respect to the input
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

dOutdWeight = cell(1,Network.numCalcs);
% dOutdBias = cell(1,Network.numCalcs);

%deltaL = exp(Z{end})-DesireOutput; % Poisson deviance cost function
                                           % with exponential output neurons
%deltaL = (Z{end}-DesireOutput); % linear output neuron, mean-squared error cost
% tmp = Output{end}; %apply softmax
% softmaxout = exp(tmp)./sum(exp(tmp));
deltaL = Output{end}(outputDim); % cross-entropy cost with sigmoid output neurons
                                % .*SigmoidPrime(Output{end}); % add this back for 
                                % mean-squared error cost function, unless
                                % you want the output neuron to be linear
dOutdWeight{end} = Activations{end}*deltaL';
% dOutdBias{end} = deltaL;

for ii=Network.numCalcs:-1:2
    if ii==Network.numCalcs
        deltaL = (Network.Weights{ii}(:,outputDim)*deltaL).*SwishPrime(Z{ii-1});
    else
        deltaL = (Network.Weights{ii}*deltaL).*SwishPrime(Z{ii-1});
    end
    
    dOutdWeight{ii-1} = Activations{ii-1}*deltaL';
%     dOutdBias{ii-1} = deltaL;
end

gradient = Network.Weights{1}*deltaL;

end