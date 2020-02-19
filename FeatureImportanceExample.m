% FeatureImportanceExample.m

load('TestData.mat');
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

load('MNIST_Network.mat','myNet');

whichIms = randperm(numImages,10);

for ii=1:10
    [featImport] = GetFeatureImport(myNet,Images(:,whichIms(ii)));
    [Output,~] = Feedforward(Images(:,whichIms(ii)),myNet);
    [~,outputDim] = max(Output{end});
    
    figure;subplot(1,2,1);
    imagesc(reshape(Images(:,whichIms(ii)),[28,28]));
    title(sprintf('Ground Truth: %d',Labels(whichIms(ii))));
    colormap('gray');
    subplot(1,2,2);
    imagesc(reshape(featImport,[28,28]));
    tmp = max(featImport);
    caxis([-tmp tmp]);title(sprintf('Model Output: %d',outputDim-1));
    colormap('gray');
end