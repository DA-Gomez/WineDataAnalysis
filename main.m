white= readtable('./files/winequality-white.csv', 'VariableNamingRule', 'preserve');
red= readtable('./files/winequality-red.csv', 'VariableNamingRule', 'preserve');

wines = cleanData(red, white);



% Wine properties
quality = wines.quality;
n_total = height(wines);
bad_quality = quality <= 5;
good_quality = quality > 5;

p_bad = sum(bad_quality) / n_total;
p_good = sum(good_quality) / n_total;

fprintf('Total: %d, Bad: %d, Good: %d\n', n_total, sum(bad_quality), sum(good_quality));
fprintf('P(bad): %.3f, P(good): %.3f\n', p_bad, p_good);

% split test on quality (needed?)
entropy_quality = -(p_bad*log2(p_bad) + p_good*log2(p_good));
gini_quality = 1 - p_bad^2 - p_good^2;

% processData(wines.alcohol, entropy_quality, good_quality, bad_quality)
%alcohol
%IG: 0.104, gini: 0.404

%density
%IG 0.044, gini: 0.440

%vol acidity
%IG: 0.029, gini: 0.449

%chlorides
%IG: 0.024, gini: 0.452

%sulphates
% IG: 0.002, gini: 0.467

%pH
%IG: 0.002, gini: 0.467

%totalSulphurDioxide
%IG: 0.001, gini: 0.467

%freeSulphurDioxide
%IG: 0.003, gini: 0.466

%residualSugar
%IG: 0.001, gini: 0.467

%citricAcid
%IG: 0.003, gini: 0.466

%fixedAcidity
%IG: 0.006, gini: 0.464








