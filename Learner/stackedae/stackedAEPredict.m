function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

disp(size(theta));

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

disp(size(softmaxTheta));
disp(hiddenSize*numClasses);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

pred = data;
for i=1:size(stack,1)
    w = stack{i}.w;
    b = stack{i}.b;
    pred = sigmoid(w*pred + repmat(b,1,size(pred,2)));
end

[~, pred] = max(softmaxTheta*pred,[],1);
end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end