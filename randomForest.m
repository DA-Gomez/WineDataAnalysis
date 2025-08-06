function out = randomForest(XTrain, YTrain, XTest, YTest)

    % Parameters
    numTrees = 500;
    numPredictorsToSample = 3;
    
    model = TreeBagger(numTrees, XTrain, YTrain, ...
        'Method', 'classification', ...
        'NumPredictorsToSample', numPredictorsToSample, ...
        'OOBPrediction', 'On');

    % Predict on Test Set
    prediction = predict(model, XTest);
    
    % Calculate Accuracy and Error Rate
    accuracy = sum(prediction == YTest) / numel(YTest);
    errorRate = 1 - accuracy;

    fprintf('Tree boostrapping Accuracy: %.4f\n', accuracy);
    fprintf('Tree boostrapping Error Rate: %.4f\n', errorRate);

    confmat = confusionmat( ... %we need them to be categorical
        categorical(YTest, categories(YTest), 'Ordinal', false), ...
        categorical(prediction, categories(YTest), 'Ordinal', false));
    disp('Confusion Matrix:');
    disp(confmat);

% Dataset random forest bootstrapping--------------------------------------
    numBootstrapSamples = 30;

    acc = zeros(numBootstrapSamples,1);
    err = zeros(numBootstrapSamples,1);
    recall = zeros(numBootstrapSamples,1);
    fallout = zeros(numBootstrapSamples,1);
    n = size(XTrain,1);

    for i = 1:numBootstrapSamples
        % Bootstrap sample (with replacement)
        idx = randsample(n, n, true);
        XBoot = XTrain(idx,:);
        YBoot = YTrain(idx);

        % Train Random Forest
        RF = TreeBagger(numTrees, XBoot, YBoot, ...
            'Method', 'classification', ...
            'NumPredictorsToSample', numPredictorsToSample, ...
            'OOBPrediction', 'Off');

        % Predict on test set
        YPred = predict(RF, XTest);
        YPred = categorical(YPred);

        % Ensure both are nominal (not ordinal)
        YTestNom = categorical(YTest, categories(YTest), 'Ordinal', false);
        YPredNom = categorical(YPred, categories(YTest), 'Ordinal', false);

        % Metrics: Accuracy and Error Rate
        acc(i) = mean(YPredNom == YTestNom);
        err(i) = 1 - acc(i);

        % Get positive/negative class labels
        cats = categories(YTestNom);
        posClass = cats{end}; % You can set this explicitly if needed
        negClass = cats{1};

        % Confusion matrix
        TP = sum((YTestNom == posClass) & (YPredNom == posClass));
        FN = sum((YTestNom == posClass) & (YPredNom == negClass));
        FP = sum((YTestNom == negClass) & (YPredNom == posClass));
        TN = sum((YTestNom == negClass) & (YPredNom == negClass));

        recall(i) = TP / (TP + FN + eps);      % Sensitivity
        fallout(i) = FP / (FP + TN + eps);     % False positive rate
    end

    % Print means only
    fprintf('Bootstrapped Accuracy: %.4f\n', mean(acc));
    fprintf('Bootstrapped Error Rate: %.4f\n', mean(err));
    fprintf('Bootstrapped Recall: %.4f\n', mean(recall));
    fprintf('Bootstrapped Fallout: %.4f\n', mean(fallout));
end
