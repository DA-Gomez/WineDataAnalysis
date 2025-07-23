function out = knn(data)
    % do knn(wines) in console to run
    data.quality = categorical(data.quality > 5, [false, true], {'bad', 'good'});
    wines = removevars(data, 'type');

    selectedVars = wines.Properties.VariableNames;
    selectedVars(strcmp(selectedVars, 'quality')) = []; %remove quality from selectedVars
    
    for i = 1:numel(selectedVars) % Min-max normalization
        col = wines.(selectedVars{i});
        min_col = min(col);
        max_col = max(col);
        wines.(selectedVars{i}) = (col - min_col) / (max_col - min_col);
    end
    
    X = table2array(wines(:, selectedVars));
    Y = wines.quality;

    %rng(1) makes it so matlab produces the same result every time we run
    %it
    rng(1);
    
    % Split into train 70% and test 30%
    cv = cvpartition(length(Y), 'HoldOut', 0.3);
    Xtrain = X(training(cv), :);
    Ytrain = Y(training(cv));
    Xtest = X(test(cv), :);
    Ytest = Y(test(cv));
    
    k = 5; %knn model using k=5
    knn = fitcknn(Xtrain, Ytrain, 'NumNeighbors', k);
    
    prediction = predict(knn, Xtest); %predict y
    
    accuracy = sum(prediction == Ytest) / length(Ytest);
    
    fprintf('Test set accuracy: %.2f%%\n', accuracy*100);

    confmat = confusionmat(Ytest, prediction); %build confusion matrix
    disp('Confusion Matrix (Rows: True class, Columns: Predicted class):');
    disp(confmat);
    
    % If you have the Statistics and Machine Learning Toolbox, you can plot it:
    figure;
    confusionchart(Ytest, prediction, ...
        'RowSummary','row-normalized', ...
        'ColumnSummary','column-normalized', ...
        'Title','Confusion Matrix for KNN');