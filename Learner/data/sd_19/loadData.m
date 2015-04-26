function [x] = loadData(parameter)
    x = load(parameter);
    x = x.(parameter);
    x = 1-(x ./ 255);
end

