function out = cleanData(red, white)
    %performs data cleaning on the data (includes removing missing values,
    %removing duplicate values and integrating the data
    
    %Remove missing values
    red = rmmissing(red);
    white = rmmissing(white);
    
    %Remove duplicate values
    red = unique(red, "rows");
    white = unique(white, "rows");

    %Add type to each data point
    red.type = repmat("red", height(red), 1);
    white.type = repmat("white", height(white), 1);
    
    %Combine the data
    out = [red; white];
end