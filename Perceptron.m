% Perceptron.m
% Created: 2016/02/05, 24 Cummington, Boston
%  Byron Price
% Updated: 2016/02/08
%   By: Byron Price

%% This code will implement a perceptron that 
%%  will be capable of recognizing handwritten text

%% Simple perceptron rule
%%   If w is a connection strength vector (or weight) 
%%   and x is an input vector, then 
%%     output = 0 if w*x + b <= 0
%%     output = 1 if w*x + b > 0 
%%   where * is the dot product of all inputs by their
%%   respective weights

%% Sigmoid perceptron
%%  If x is an input vector and w a connection strength
%%  vector, then output = 1/(1+exp(-(w*x+b)))
%%
%% See www.neuralnetworksanddeeplearning.com for more information.

load('TrainingData.mat')

numImages = size(Images,2);
numPixels = size(Images,1);

% CREATE THE NETWORK WITH RANDOMIZED WEIGHTS AND BIASES
numDigits = 10;
numHidden = 15;
myNet = Network([numPixels,numHidden,numDigits]); % from a function
     % in this directory, builds a 3-layer network
     
DesireOutput = zeros(numDigits,numImages);

for ii=1:numImages
    numVector = zeros(numDigits,1);
    for jj=1:numDigits
        if Labels(ii) == jj-1
            numVector(jj) = 1;
            DesireOutput(:,ii) = numVector;
        end
    end
end

% STOCHASTIC GRADIENT DESCENT
batchSize = 10; % make mini batches and run the algorithm
     % on those "runs" times
runs = 100000;
eta = 3;

for ii=1:runs
    indeces = ceil(rand([batchSize,1]).*(numImages-1));
    dCostdWeight = cell(1,2);
    dCostdBias = cell(1,2);
    dCostdWeight{1} = zeros(numPixels,numHidden);
    dCostdWeight{2} = zeros(numHidden,numDigits);
    dCostdBias{1} = zeros(numHidden,1);
    dCostdBias{2} = zeros(numDigits,1);
    
    for jj=1:batchSize
        index = indeces(jj);
        [costweight,costbias] = BackProp(Images(:,index),myNet,...
        DesireOutput(:,index));
        for kk=1:2
            dCostdWeight{kk} = dCostdWeight{kk}+costweight{kk};
            dCostdBias{kk} = dCostdBias{kk}+costbias{kk};
        end
    end
    [myNet] = GradientDescent(myNet,dCostdWeight,dCostdBias,batchSize,eta);
    clear indeces dCostdWeight dCostdBias;
end

% COMPARE ON TEST DATA
clear Images Labels;
load('TestData.mat')
numImages = size(Images,2);
numPixels = size(Images,1);
numDigits = 10;

DesireOutput = zeros(numDigits,numImages);

for ii=1:numImages
    numVector = zeros(numDigits,1);
    for jj=1:numDigits
        if Labels(ii) == jj-1
            numVector(jj) = 1;
            DesireOutput(:,ii) = numVector;
        end
    end
end

classifiedVals = zeros(numImages,1);
count = 0;
for ii=1:numImages
[Output,Z] = Feedforward(Images(:,ii),myNet);
[~,realVal] = max(DesireOutput(:,ii));
[~,netVal] = max(Output{2});
classifiedVals(ii) = netVal-1;
if realVal == netVal
count = count+1;
end
end
Accuracy = count/numImages;
