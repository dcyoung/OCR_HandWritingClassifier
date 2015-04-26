function [y] = loadLabels(parameter)
    y = load(parameter)';
    y = y.(parameter)';
    [~,~,y] = unique(y);
end

