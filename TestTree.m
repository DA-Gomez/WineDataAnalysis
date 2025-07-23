red = readtable('winequality-red.csv', 'VariableNamingRule', 'preserve');
white = readtable('winequality-white.csv', 'VariableNamingRule', 'preserve');

red.type = repmat("red", height(red), 1);
white.type = repmat("white", height(white), 1);

data = [red; white];

features = data{:, 1:end-2}; 
labels = data.quality;

% our range would be from 5<= 0 which is bad and 5> which is good
classLabels = categorical(labels > 5, [false, true], {'bad', 'good'});

%60% train, 20% validation, 20% test
cv = cvpartition(classLabels, 'HoldOut', 0.4);
Xtrain = features(training(cv), :);
Ytrain = classLabels(training(cv));
Xtemp = features(test(cv), :);
Ytemp = classLabels(test(cv));

cv2 = cvpartition(Ytemp, 'HoldOut', 0.5);
Xval = Xtemp(training(cv2), :);
Yval = Ytemp(training(cv2));
Xtest = Xtemp(test(cv2), :);
Ytest = Ytemp(test(cv2));

varNames = data.Properties.VariableNames(1:end-2);
XtrainTbl = array2table(Xtrain, 'VariableNames', varNames);
XvalTbl   = array2table(Xval,   'VariableNames', varNames);
XtestTbl  = array2table(Xtest,  'VariableNames', varNames);


treeModel = fitctree(XtrainTbl, Ytrain, ...
    'MaxNumSplits', 20, ... %by changing the value of max and min we can control the pruning and leaf size
    'MinLeafSize', 25, ...
    'Prune', 'on');

validLevels = 0:(numel(treeModel.PruneAlpha) - 1);
valAcc = zeros(size(validLevels));
numNodes = zeros(size(validLevels));

for i = 1:length(validLevels)
    pruned = prune(treeModel, 'Level', validLevels(i));
    YvalPred = predict(pruned, XvalTbl);
    valAcc(i) = mean(YvalPred == Yval);
    numNodes(i) = pruned.NumNodes;
end

% try to find the highest accuracy 
[bestAcc, bestIdx] = max(valAcc);
threshold = 0.90 * bestAcc;
acceptableIdx = find(valAcc >= threshold);
[~, simplestIdx] = min(numNodes(acceptableIdx));
bestLevel = validLevels(acceptableIdx(simplestIdx));

prunedTree = prune(treeModel, 'Level', bestLevel);

YtestPred = predict(prunedTree, XtestTbl);
testAccuracy = mean(YtestPred == Ytest);
disp("Final test accuracy: " + testAccuracy);
disp("Chosen pruning level: " + bestLevel + " (Nodes: " + prunedTree.NumNodes + ")");

% to be honest whenever you want you can remove this guys cause i just
% made this to check the process and progress so it's not required
fprintf('\nLevel | Accuracy | Nodes\n');
for i = 1:length(validLevels)
    fprintf('  %2d   |  %.3f   |  %3d\n', validLevels(i), valAcc(i), numNodes(i));
end

view(prunedTree, 'Mode', 'graph');
