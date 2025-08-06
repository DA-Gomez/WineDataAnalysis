function out = NewTree()
    rng(1);
    
    red = readtable('winequality-red.csv', 'VariableNamingRule', 'preserve');
    white = readtable('winequality-white.csv', 'VariableNamingRule', 'preserve');
    
    red.type = repmat("red", height(red), 1);
    white.type = repmat("white", height(white), 1);
    
    data = [red; white];
    labels = data.quality;
    
    excellentData = data(labels >= 8, :);
    upsampledExcellent = repmat(excellentData, 5, 1);
    data_augmented = [data; upsampledExcellent];
    
    features = data_augmented{:, 1:end-2};
    labels = data_augmented.quality;
    
    classLabels = repmat("poor", height(data_augmented), 1);
    classLabels(labels >= 6 & labels <= 7) = "normal";
    classLabels(labels >= 8) = "excellent";
    classLabels = categorical(classLabels);
    
    cv = cvpartition(height(data_augmented), 'HoldOut', 0.3);
    trainIdx = training(cv);
    testIdx = test(cv);
    
    Xtrain = features(trainIdx, :);
    Ytrain = classLabels(trainIdx);
    Xtest = features(testIdx, :);
    Ytest = classLabels(testIdx);
    
    treeModel = fitctree(Xtrain, Ytrain, ...
        'PredictorNames', data_augmented.Properties.VariableNames(1:end-2), ...
        'MaxNumSplits', 7, ...
        'MinLeafSize', 30, ...
        'Prune', 'off');
    
    Ypred = predict(treeModel, Xtest);
    
    accuracy = sum(Ypred == Ytest) / numel(Ytest);
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);
    
    confMat = confusionmat(Ytest, Ypred);
    disp('Confusion Matrix:');
    disp(confMat);
    
    confusionchart(Ytest, Ypred);
    
    cvModel = crossval(treeModel, 'KFold', 5);
    cvLoss = kfoldLoss(cvModel);
    fprintf('5-Fold Cross-Validation Loss: %.4f\n', cvLoss);
    fprintf('Cross-Validation Accuracy: %.2f%%\n', (1 - cvLoss) * 100);
    
    view(treeModel, 'Mode', 'graph');
