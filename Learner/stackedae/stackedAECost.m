function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
stackgrad = cell(size(stack));

% You might find these variables useful
m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

layers = numel(stack) + 1;

a = cell(layers,1);
a{1} = data;
for i = 2:layers
    w = stack{i-1}.w;
    b = stack{i-1}.b; 
    a{i} = sigmoid(w*a{i-1} + repmat(b,1,size(a{i-1},2)));
end

P = softmaxTheta*a{layers};
expM = exp(bsxfun(@minus, P, max(P,[],1)));
h = bsxfun(@rdivide, expM, sum(expM));

expP = exp(bsxfun(@minus, P, h));
innerTerm = groundTruth-bsxfun(@rdivide, expP, sum(expP));

softmaxThetaGrad = -((groundTruth - h)*a{layers}')/m;

delta = cell(layers,1);
delta{layers} = -(softmaxTheta'*innerTerm).*(a{layers}.*(1-a{layers}));
for i = layers-1:-1:2
    w = stack{i}.w;
    delta{i} = (w'*delta{i+1}).*(a{i}.*(1-a{i}));
end

wsum = 0;
for d = 1:layers-1
    w = stack{d}.w;
    stackgrad{d}.w = (delta{d+1}*a{d}')/m + lambda*w;
    stackgrad{d}.b = sum(delta{d+1},2)/m;
    
    wsum = wsum + sum(sum(w.^2));
end

cost = -sum(sum(groundTruth.*log(h)))/m + (lambda/2)*wsum;

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
