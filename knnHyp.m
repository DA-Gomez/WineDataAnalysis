function out = knnHyp(XTrain, YTrain, XTest, YTest)
    k_values = 1:2:21; % odd k from 1 to 21
    best_k = k_values(1);
    best_acc = 0;
    
    for k = k_values
        model = fitcknn(XTrain, YTrain, 'NumNeighbors', k);
        pred = predict(model, XTest);
        acc = sum(pred == YTest) / length(YTest);
        fprintf('k = %d, Accuracy = %.2f%%\n', k, acc * 100);
        
        if acc > best_acc
            best_acc = acc;
            best_k = k;
        end
    end
    
    fprintf('Best k: %d with Accuracy = %.2f%%\n', best_k, best_acc * 100);
    
    finalModel = fitcknn(XTrain, YTrain, 'NumNeighbors', best_k);
    finalPred = predict(finalModel, XTest);
    
    confmat = confusionmat(YTest, finalPred);
    disp('Confusion Matrix (Rows: True class, Columns: Predicted class):');
    disp(confmat);
