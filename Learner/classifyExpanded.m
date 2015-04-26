function classifyExpanded(dataset)
    addpath data
    if exist(strcat('data/', dataset),'dir') == 0
        fprintf('Data set does not exist.\n');
        return;
    end
    [~,~,xTe,yTe] = prep_data(dataset);
    load(strcat(dataset, '_classifier'));
    rmpath data
    
    lookupTable = [ ...
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ...
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', ...
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', ...
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', ...
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', ...
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', ...
    'y', 'z' ];
    
    addpath stackedae
    [pred] = stackedAEPredict(classifier.theta, classifier.inputSize, ...
                              classifier.hiddenSize, classifier.numClasses, ...
                              classifier.netconfig, xTe);
    rmpath stackedae
    
    misclassified = find(pred(:) ~= yTe(:));
    disp(size(misclassified));
    for i=1:size(misclassified,1)
        sample = xTe(:,misclassified(i));
        disp(size(sample));
        sample = reshape(sample, [], 32);
        sample = fliplr(rot90(sample,3));
        imshow(sample);
        trueLabel = lookupTable(yTe(misclassified(i)));
        assignedLabel = lookupTable(pred(misclassified(i)));
        fprintf('True label: %c\tAssigned label: %c\n', trueLabel, assignedLabel);
        
        waitforbuttonpress;
    end
end