function [xTr, yTr, xTe, yTe] = prep_data(dataset)

if exist(strcat('data/', dataset),'dir') == 0 && exist(dataset,'dir') == 0
    fprintf('Data set does not exist.\n');
    xTr = 0; yTr = 0; xTe = 0; yTe = 0;
    return;
end

pathName = strcat(fileparts(which('prep_data.m')), '/', dataset);
addpath(pathName);

xTr = double(loadData('trainX'));
yTr = double(loadLabels('trainY'));

xTe = double(loadData('testX'));
yTe = double(loadLabels('testY'));

rmpath(pathName);

end

