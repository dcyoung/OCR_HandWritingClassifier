function train(dataset, options)
    if nargin < 2; options = []; end
    
    %==========================================================
    %% LOAD TRAINING DATA
    addpath data
    [xTr, yTr] = prep_data(dataset);
    rmpath data
    
    %==========================================================
    %% TRAIN A CLASSIFIER ON THE DATA
    classifier = trainClassifier(xTr, yTr, options);

    %==========================================================
    %% TEST THE ACCURACY & COMPARE AGAINST ANY OLDER CLASSIFIER
    if exist(strcat(dataset, '_classifier.mat'), 'file') ~= 0
        [acc, time] = classify(dataset);
        fprintf('Prior error rate: %0.2f%% in %.0f milliseconds\n', acc, time);
    end
    save(strcat(dataset, '_classifier'), 'classifier');
    [acc, time] = classify(dataset);
    fprintf('Error rate: %0.2f%% in %.0f milliseconds\n', acc, time);
    
end

