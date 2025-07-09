function out = tree3(wines)

classLabels = categorical(wines.quality > 5, [false, true], {'bad', 'good'});

selectedVars = {'alcohol', 'density', 'volatileAcidity', 'chlorides', 'fixedAcidity'};

features = wines{:, selectedVars};

treeModel = fitctree(features, classLabels, ...
    'PredictorNames', selectedVars, ...
    'MaxNumSplits', 12, ...
    'MinLeafSize', 25);

view(treeModel, 'Mode', 'graph');

% [predictedLabels, nodeNumbers] = predict(treeModel, features);

% labels = wines.quality;  % Numeric wine quality (0–10)
% 
% % regression decision tree
% regTree = fitrtree(features, labels, ...
%     'PredictorNames', selectedVars, ...
%     'MaxNumSplits', 12, ...
%     'MinLeafSize', 25);
% 
% view(regTree, 'Mode', 'graph');
