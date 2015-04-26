function [x] = loadData(parameter)
    x = load(parameter);
    x = x.(parameter);
    x = x ./ 255;
end

