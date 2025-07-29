function out = randomForest(XTrain, YTrain, XTest, YTest)

    % Parameters
    numTrees = 500;
    numPredictorsToSample = 3;
    
    model = TreeBagger(numTrees, XTrain, YTrain, ...
        'Method', 'classification', ...
        'NumPredictorsToSample', numPredictorsToSample, ...
        'OOBPrediction', 'On');  % Optionally include this line

    % Predict on Test Set
    YPred = predict(model, XTest);
    

    % Calculate Accuracy and Error Rate
    accuracy = sum(YPred == YTest) / numel(YTest);
    errorRate = 1 - accuracy;

    fprintf('Tree boostrapping Accuracy: %.4f\n', accuracy);
    fprintf('Tree boostrapping Error Rate: %.4f\n', errorRate);


% Dataset random forest bootstrapping--------------------------------------
    % numBootstrapSamples = 30;
    % 
    % acc = zeros(numBootstrapSamples,1);
    % err = zeros(numBootstrapSamples,1);
    % n = size(XTrain,1);
    % 
    % for i = 1:numBootstrapSamples
    %     % Bootstrap sample (with replacement)
    %     idx = randsample(n, n, true);
    %     XBoot = XTrain(idx,:);
    %     YBoot = YTrain(idx);
    % 
    %     % Train Random Forest
    %     RF = TreeBagger(numTrees, XBoot, YBoot, ...
    %         'Method', 'classification', ...
    %         'NumPredictorsToSample', numPredictorsToSample, ...
    %         'OOBPrediction', 'Off');
    % 
    %     % Predict on test set
    %     YPred = predict(RF, XTest);
    %     YPred = categorical(YPred);
    % 
    %     % Metrics
    %     YTest = categorical(YTest, categories(YTest), 'Ordinal', false);
    %     YPred = categorical(YPred, categories(YTest), 'Ordinal', false);
    % 
    %     acc(i) = mean(YPred == YTest);
    %     err(i) = 1 - acc(i);
    % end
    % 
    % % Aggregate results
    % 
    % fprintf('Dataset boostrapping Accuracy: %.4f\n', mean(acc));
    % fprintf('Dataset boostrapping Error Rate: %.4f\n', mean(err));
    % 
    % out.meanAccuracy = mean(acc);
    % out.stdAccuracy = std(acc);
    % out.meanErrorRate = mean(err);
    % out.stdErrorRate = std(err);
    % out.allAccuracies = acc;
    % out.allErrorRates = err;
end
