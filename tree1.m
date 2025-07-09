function out = tree1(wines)

%raw tree no pruning
data = wines;
data.quality = categorical(data.quality > 5, [false, true], {'bad', 'good'});

%exclude quality from tree
features = data(:, setdiff(data.Properties.VariableNames, {'quality'}));

tree = fitctree(features, data.quality);

view(tree, "Mode","graph")