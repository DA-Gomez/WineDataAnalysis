white= readtable('./files/winequality-white.csv', 'VariableNamingRule', 'preserve');
red= readtable('./files/winequality-red.csv', 'VariableNamingRule', 'preserve');

wines = cleanData(red, white);



% % Wine properties
% quality = wines.quality;
% n_total = height(wines);
% bad_quality = quality <= 5;
% good_quality = quality > 5;
% 
% p_bad = sum(bad_quality) / n_total;
% p_good = sum(good_quality) / n_total;
% 
% fprintf('Total: %d, Bad: %d, Good: %d\n', n_total, sum(bad_quality), sum(good_quality));
% fprintf('P(bad): %.3f, P(good): %.3f\n', p_bad, p_good);
% 
% % split test on quality (needed?)
% entropy_quality = -(p_bad*log2(p_bad) + p_good*log2(p_good));
% gini_quality = 1 - p_bad^2 - p_good^2;
% 

fprintf('\nAlcohol:\n')
processData(wines.alcohol, wines.quality)

fprintf('\nDensity:\n')
processData(wines.density, wines.quality)

fprintf('\nvol acidity:\n')
processData(wines.("volatile acidity"), wines.quality)

fprintf('\nchlorides:\n')
processData(wines.chlorides, wines.quality)

fprintf('\nsulphates:\n')
processData(wines.sulphates, wines.quality)

%pH
fprintf('\npH:\n')
processData(wines.pH, wines.quality)
%IG: 0.002, gini: 0.467

%totalSulphurDioxide
fprintf('\ntotalSulphurDioxide:\n')
processData(wines.("total sulfur dioxide"), wines.quality)
%IG: 0.001, gini: 0.467

%freeSulphurDioxide
fprintf('\nfreeSulphurDioxide:\n')
processData(wines.("free sulfur dioxide"), wines.quality)
%IG: 0.003, gini: 0.466

%residualSugar
fprintf('\nresidualSugar:\n')
processData(wines.("residual sugar"), wines.quality)
%IG: 0.001, gini: 0.467

%citricAcid
fprintf('\ncitricAcid:\n')
processData(wines.("citric acid"), wines.quality)

%fixedAcidity
fprintf('\nfixedAcidity:\n')
processData(wines.("fixed acidity"), wines.quality)

edges = [3, 5, 7, 9]; % Define the edges like R did
labels = {'low', 'medium', 'high'}; % Custom labels
wines.quality = discretize(wines.quality, edges, 'categorical', labels);
wines = removevars(wines, 'type');

rng(123);
X = wines{:, 1:end-1}; % All columns except quality
Y = wines.quality;

X = (X - min(X)) ./ (max(X) - min(X));

% Split into train 80% and test 20
cv = cvpartition(Y, 'HoldOut', 0.2);
XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);
XTest  = X(test(cv), :);
YTest  = Y(test(cv), :);

hold on
fprintf("\nKNN metrics: \n")
knn(XTrain, YTrain, XTest, YTest);
fprintf("\nRandom Forest metrics: \n")
randomForest(XTrain, YTrain, XTest, YTest);
hold off