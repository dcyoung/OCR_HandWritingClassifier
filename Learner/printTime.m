function printTime(time)
    %==========================================================
    %% PRINT OUT THE TIME
    %  current if 0 args, specified if 1 arg
    if nargin == 0, time = clock; end
    fprintf('%02d:%02d:%02d on %02d/%02d/%04d\n', ...
        time(4), time(5), floor(time(6)), time(2), time(3), time(1));
end