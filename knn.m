function out = knn(XTrain, YTrain, XTest, YTest)
    % do knn(wines) in console to run
    
    k = 5; %knn model using k=5
    knn = fitcknn(XTrain, YTrain, 'NumNeighbors', k);
    
    prediction = predict(knn, XTest); %predict y
    
    accuracy = sum(prediction == YTest) / length(YTest);
     
    fprintf('Test set accuracy: %.2f%%\n', accuracy*100);

    confmat = confusionmat(YTest, prediction); %build confusion matrix
    disp('Confusion Matrix (Rows: True class, Columns: Predicted class):');
    disp(confmat);
    
    figure;
    confusionchart(YTest, prediction, ...
        'RowSummary','row-normalized', ...
        'ColumnSummary','column-normalized', ...
        'Title','Confusion Matrix for KNN');