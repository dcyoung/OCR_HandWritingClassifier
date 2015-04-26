function [err, ms] = classify(dataset)

    %==========================================================
    %% LOAD TESTING DATA & CLASSIFIER
    addpath data
    if exist(strcat('data/', dataset),'dir') == 0
        fprintf('Data set does not exist.\n');
        return;
    end
    [~,~,xTe,yTe] = prep_data(dataset);
    load(strcat(dataset, '_classifier'));
    rmpath data
    
    %==========================================================
    %% MAKE PREDICTIONS ON THE DATA & COMPUTE ACCURACY
    addpath stackedae
    tic
    [pred] = stackedAEPredict(classifier.theta, classifier.inputSize, ...
                              classifier.hiddenSize, classifier.numClasses, ...
                              classifier.netconfig, xTe);
    acc = mean(yTe(:) == pred(:));
    err = 100 - (acc * 100);
    ms = toc*1000;
    rmpath stackedae
end