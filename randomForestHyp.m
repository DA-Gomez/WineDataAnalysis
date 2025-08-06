function out = randomForestHyp(XTrain, YTrain, XTest, YTest)
    numTreesGrid = [100, 300, 500];
    numPredictorsGrid = [2, 3, 4];

    bestAcc = 0;
    bestPred = [];

    for nt = numTreesGrid
        for np = numPredictorsGrid
            model = TreeBagger(nt, XTrain, YTrain, ...
                'Method', 'classification', ...
                'NumPredictorsToSample', np, ...
                'OOBPrediction', 'On');

            prediction = predict(model, XTest);
            acc = sum(categorical(prediction) == categorical(YTest)) / numel(YTest);

            fprintf('numTrees: %d, numPredictors: %d, Accuracy: %.4f\n', ...
                nt, np, acc);

            if acc > bestAcc %find best model
                bestAcc = acc;
                bestPred = prediction;
            end
        end
    end

    % Display confusion matrix for best model
    disp('Best model confusion matrix (hyperparameter)');
    confmat = confusionmat(categorical(YTest), categorical(bestPred));
    disp(confmat);

    out = struct('BestAccuracy', bestAcc, 'ConfusionMatrix', confmat);