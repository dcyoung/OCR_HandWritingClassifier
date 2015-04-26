function organize()
%==========================================================
%% ORGANIZE
% This function, when run in a directory containing NIST's SD-19
% dataset, will open and read all of the images into one of two
% large matrices: a 1024xn training set and a 1024xm test set
% (where m and n are random values based on the probability of a
% given writer being assigned to one set or the other)

topLevel = dir;

trainX = [];
testX = [];

trainY = [];
testY = [];

writerCount = 0;

for i=1:size(topLevel)
    if strfind(topLevel(i).name, 'HSF')
        midLevel = dir(topLevel(i).name);
        for j=1:size(midLevel)
            if (j >= 4)
                fprintf('Processing writer %04d\n', writerCount);
                writerCount = writerCount + 1;
            end
            lowLevel = dir(strcat(topLevel(i).name, '/', midLevel(j).name));
            testSet = rand > 5/6; % Assign a percentage of the writers to the test set
            for k=1:size(lowLevel)
                if strfind(lowLevel(k).name, 'bmp')
                    filename = strcat(topLevel(i).name, '/', midLevel(j).name, '/', lowLevel(k).name);
                    imgX = rgb2gray(imread(filename));
                    imgX = imresize(imgX, 0.25);
                    imgX = reshape(imgX.',[],1);
                    imgY = filename(1,length(filename)-7);
                    if (testSet)
                        testX = [testX imgX];
                        testY = [testY imgY];
                    else
                        trainX = [trainX imgX];
                        trainY = [trainY imgY];
                    end
                end
            end
            if (j >= 4) 
                fprintf('\tTraining size: %06d; Test size: %06d\n\n', size(trainX,2), size(testX,2));
            end
        end
    end
end

save('trainX','trainX');
save('trainY','trainY');
save('testX','testX');
save('testY','testY');

percentTrain = size(trainX,2) / size([trainX testX],2);
percentTest = 1 - percentTrain;

fprintf('%0.3f%% of the data is for training; %0.3f%% is for testing.\n', percentTrain, percentTest);

end
