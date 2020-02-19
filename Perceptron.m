% Perceptron.m
% Created: 2016/02/05, 24 Cummington, Boston
%  Byron Price
% Updated: 2016/10/13
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
numHidden1 = 100;
myNet = Network([numPixels,numHidden1,50,numDigits]); % from a function
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
runs = 1e4;
eta = 0.01; % learning rate
lambda = 10; % L2 regularization parameter
alpha = 0.75; % proportion of hidden nodes to keep during dropout

numCalcs = myNet.numCalcs;
dCostdWeight = cell(1,numCalcs);
dCostdBias = cell(1,numCalcs);

for ii=1:runs
    indeces = ceil(rand([batchSize,1]).*(numImages-1));
    [dropOutNet,dropOutInds] = MakeDropOutNet(myNet,alpha);

    for jj=1:numCalcs
        dCostdWeight{jj} = zeros(size(dropOutNet.Weights{jj}));
        dCostdBias{jj} = zeros(size(dropOutNet.Biases{jj}));
    end
    for jj=1:batchSize
        index = indeces(jj);
        [costweight,costbias] = BackProp(Images(:,index),dropOutNet,...
        DesireOutput(:,index));
        for kk=1:numCalcs
            dCostdWeight{kk} = dCostdWeight{kk}+costweight{kk};
            dCostdBias{kk} = dCostdBias{kk}+costbias{kk};
        end
    end
    [dropOutNet] = GradientDescent(dropOutNet,dCostdWeight,dCostdBias,batchSize,eta,numImages,lambda);
    [myNet] = RevertToWholeNet(dropOutNet,myNet,dropOutInds);
%     clear indeces;% dCostdWeight dCostdBias;
end

[myNet] = AdjustDropOutNet(myNet,alpha);

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
    [~,netVal] = max(Output{end});
    classifiedVals(ii) = netVal-1;
    if realVal == netVal
        count = count+1;
    end
end
Accuracy = count/numImages;

fprintf('Accuracy: %3.3f\n',Accuracy);

% for ii=1:5
%     index = ceil(rand*(numImages-1));
%     digit = classifiedVals(index);
%     image = reshape(Images(:,index),[28,28]);
%     figure();imagesc(image);title(sprintf('Classified as a(n) %i',digit));
%     colormap gray;
% end
