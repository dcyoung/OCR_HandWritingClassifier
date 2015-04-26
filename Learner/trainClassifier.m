function classifier = trainClassifier(xTr, yTr, options)

if nargin < 3; options = []; end

inputSize = size(xTr,1);
numClasses = numel(unique(yTr));

%==========================================================
%% DEFAULT VALUES

% NETWORK STRUCTURE
hiddenLayersDEFAULT  = 2;
hiddenSizeDEFAULT    = 500;

% PARAMETERS
sparsityDEFAULT      = 0.1;
lambdaDEFAULT        = 3e-3;
betaDEFAULT          = 3;

% STEP SIZES
featStepsDEFAULT     = 200;
softmaxStepsDEFAULT  = 400;
fineTuneStepsDEFAULT = 400;

%==========================================================
%% Add the requisite paths

addpath autoencoder
addpath stackedae
addpath softmax
addpath minFunc

%==========================================================
%% Set all parameters

% hidden layer count
if isfield(options, 'hiddenLayers')
    hiddenLayers = options.hiddenLayers; else hiddenLayers = hiddenLayersDEFAULT;
end

% hidden layer size
if isfield(options, 'hiddenSize')
    hiddenSize = options.hiddenSize; else hiddenSize = hiddenSizeDEFAULT;
end

% desired average activation of the hidden units.
if isfield(options, 'sparsity')
    sparsityParam = options.sparsity; else sparsityParam = sparsityDEFAULT;
end

% weight decay parameter
if isfield(options, 'lambda')
    lambda = options.lambda; else lambda = lambdaDEFAULT;
end

% weight of sparsity penalty term
if isfield(options, 'beta')
    beta = options.beta; else beta = betaDEFAULT;
end

% number of gradient descent steps per autoencoder
if isfield(options, 'featSteps')
    featSteps = options.featSteps; else featSteps = featStepsDEFAULT;
end

% number of gradient descent steps for softmax regression
if isfield(options, 'softmaxSteps')
    softmaxSteps = options.softmaxSteps; else softmaxSteps = softmaxStepsDEFAULT;
end

% number of gradient descent steps for fine tuning
if isfield(options, 'fineTuneSteps')
    fineTuneSteps = options.fineTuneSteps; else fineTuneSteps = fineTuneStepsDEFAULT;
end

aeThetas = cell(hiddenLayers,1);
aeFeatures = cell(hiddenLayers,1);

%==========================================================
%% Train the first autoencoder layer

startTime = clock;

fprintf('Starting to train layer 1 at ');
printTime;

aeThetas{1} = initializeParameters(hiddenSize, inputSize);

options.Method = 'lbfgs';
options.maxIter = featSteps;
options.display = 'on';

aeThetas{1} = minFunc( @(p) sparseAutoencoderCost(p, ...
                              inputSize, hiddenSize, ...
                              lambda, sparsityParam, ...
                              beta, xTr), ...
                              aeThetas{1}, options);
                           
aeFeatures{1} = feedForwardAutoencoder(aeThetas{1}, hiddenSize, ...
                              inputSize, xTr);
 
weights = reshape(aeThetas{1}(1:hiddenSize*inputSize), hiddenSize, inputSize);
displayFeatures(weights', 'features_1');

%==========================================================
%% Train subsequent autoencoders

for i=2:hiddenLayers
    fprintf('Starting to train layer %i at ', i);
    printTime;
    
    aeThetas{i} = initializeParameters(hiddenSize, hiddenSize);

    options.Method = 'lbfgs';
    options.maxIter = featSteps;
    options.display = 'on';

    aeThetas{i} = minFunc( @(p) sparseAutoencoderCost(p,  ...
                                  hiddenSize, hiddenSize, ...
                                  lambda, sparsityParam,  ...
                                  beta, aeFeatures{i-1}), ...
                                  aeThetas{i}, options);

    aeFeatures{i} = feedForwardAutoencoder(aeThetas{i}, hiddenSize, ...
                                  hiddenSize, aeFeatures{i-1}); 
end

%==========================================================
%% Train the softmax layer

fprintf('Starting to train softmax layer at ');
printTime;

options.maxIter = softmaxSteps;
softmaxModel = softmaxTrain(hiddenSize, numClasses, lambda, ...
                            aeFeatures{hiddenLayers}, yTr, options);
theta = softmaxModel.optTheta(:);

%==========================================================
%% Fine tune the model

fprintf('Starting to fine tune the model at ');
printTime;

% Initialize the stack using the parameters learned
stack = cell(hiddenLayers,1);
aeTheta = aeThetas{1};
stack{1}.w = reshape(aeTheta(1:hiddenSize*inputSize), ...
                     hiddenSize, inputSize);
stack{1}.b = aeTheta(2*hiddenSize*inputSize+1:2*hiddenSize*inputSize+hiddenSize);
for i=2:hiddenLayers
    aeTheta = aeThetas{i};
    stack{i}.w = reshape(aeTheta(1:hiddenSize*hiddenSize), ...
                         hiddenSize, hiddenSize);
    stack{i}.b = aeTheta(2*hiddenSize*hiddenSize+1:2*hiddenSize*hiddenSize+hiddenSize);
end

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
theta = [theta; stackparams];

options.Method = 'lbfgs';
options.maxIter = fineTuneSteps;
options.display = 'on';


[theta, ~] = minFunc( @(p) stackedAECost(p, ...
                                inputSize, hiddenSize, ...
                                numClasses, netconfig, ...
                                lambda, xTr, yTr), ...
                                theta, options);
                            
classifier.theta = theta;
classifier.inputSize = inputSize;
classifier.hiddenSize = hiddenSize;
classifier.numClasses = numClasses;
classifier.netconfig = netconfig;

fprintf('Training complete.\n');

fprintf('Time started: ');
printTime(startTime);
fprintf('Time ended:   ');
printTime;

%==========================================================
%% Remove the requisite paths

rmpath autoencoder
rmpath stackedae
rmpath softmax
rmpath minFunc

end