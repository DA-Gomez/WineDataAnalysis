data = readtable('wine.csv');

rawY = data.quality;
Y = strings(size(rawY));
Y(rawY <= 4) = "poor";
Y(rawY >= 5 & rawY <= 6) = "normal";
Y(rawY >= 7) = "excellent";
Y = categorical(Y);

X = removevars(data, 'quality');
X = normalize(table2array(X));

poor_idx = rawY <= 4;
normal_idx = rawY >= 5 & rawY <= 6;
excellent_idx = rawY >= 7;

X_poor = X(poor_idx, :);       Y_poor = Y(poor_idx);
X_normal = X(normal_idx, :);   Y_normal = Y(normal_idx);
X_excellent = X(excellent_idx, :); Y_excellent = Y(excellent_idx);

target_size = 100;

X_poor = repmat(X_poor, ceil(target_size / size(X_poor,1)), 1);
Y_poor = repmat(Y_poor, ceil(target_size / size(Y_poor,1)), 1);

X_normal = repmat(X_normal, ceil(target_size / size(X_normal,1)), 1);
Y_normal = repmat(Y_normal, ceil(target_size / size(Y_normal,1)), 1);

X_excellent = repmat(X_excellent, ceil(target_size / size(X_excellent,1)), 1);
Y_excellent = repmat(Y_excellent, ceil(target_size / size(Y_excellent,1)), 1);

X = [X_poor(1:target_size,:); X_normal(1:target_size,:); X_excellent(1:target_size,:)];
Y = [Y_poor(1:target_size); Y_normal(1:target_size); Y_excellent(1:target_size)];

classes_to_test = ["poor", "normal", "excellent"];
B = 100;
k = 5;

for i = 1:length(classes_to_test)
    test_label = classes_to_test(i);
    test_point = X(find(Y == test_label, 1), :);

    predictions = categorical(strings(B,1));

    for b = 1:B
        idx = randsample(size(X,1), size(X,1), true);
        Xb = X(idx, :);
        Yb = Y(idx);

        X_min = min(Xb);
        X_max = max(Xb);
        Xb_norm = (Xb - X_min) ./ (X_max - X_min);
        tp_norm = (test_point - X_min) ./ (X_max - X_min);

        distances = sqrt(sum((Xb_norm - tp_norm).^2, 2));
        [~, sorted_indices] = sort(distances);
        nearest_labels = Yb(sorted_indices(1:k));
        predictions(b) = mode(nearest_labels);
    end

    final_prediction = mode(predictions);
    fprintf('\n=== Test Sample from Class: %s ===\n', test_label);
    fprintf('Bootstrap KNN final prediction (mode of %d runs): %s\n', B, string(final_prediction));
    tabulate(cellstr(predictions))
end
