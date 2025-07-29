function out = processData(attribute, quality) 
    %Processes the data, getting its values for analysis and model
    %construction by its split 
    
    entropy = @(good, mid, bad) -(good*log2(good) + mid*log2(mid) + bad*log2(bad));
    gini = @(good, mid, bad) 1 - good^2 - mid^2 - bad^2;
    
    n_total = height(attribute);
    
    % Split on attribute mean
    att_mean = mean(attribute);
    low = attribute <= att_mean;
    high = attribute > att_mean;
    
    n_low = sum(low);
    n_high = sum(high);
    
    % Define 3 classes based on quality
    is_bad = quality >= 3 & quality <= 5;  % class 1
    is_mid = quality >= 6 & quality <= 7;  % class 2
    is_good = quality >= 8 & quality <= 9; % class 3
    
    % Quality splits for low group
    low_bad = sum(low & is_bad);
    low_mid = sum(low & is_mid);
    low_good = sum(low & is_good);
    
    % Quality splits for high group
    high_bad = sum(high & is_bad);
    high_mid = sum(high & is_mid);
    high_good = sum(high & is_good);
    
    % Proportions of groups
    p_low = n_low / n_total;
    p_high = n_high / n_total;
    
    % Proportions within each group
    p_low_bad = low_bad / n_low;
    p_low_mid = low_mid / n_low;
    p_low_good = low_good / n_low;
    
    p_high_bad = high_bad / n_high;
    p_high_mid = high_mid / n_high;
    p_high_good = high_good / n_high;
    % 
    % fprintf('Mean: %.3f\n', att_mean);
    % fprintf('Below Mean: %d (%.3f), Above Mean: %d (%.3f)\n', n_low, p_low, n_high, p_high);
    % fprintf('P(low & bad): %.3f, P(low & mid): %.3f, P(low & good): %.3f\n', p_low_bad, p_low_mid, p_low_good);
    % fprintf('P(high & bad): %.3f, P(high & mid): %.3f, P(high & good): %.3f\n\n', p_high_bad, p_high_mid, p_high_good);
    
    % Entropy
    entropy_low = entropy(p_low_good, p_low_mid, p_low_bad);
    entropy_high = entropy(p_high_good, p_high_mid, p_high_bad);
    
    total_bad = sum(is_bad);
    total_mid = sum(is_mid);
    total_good = sum(is_good);
    
    p_total_bad = total_bad / n_total;
    p_total_mid = total_mid / n_total;
    p_total_good = total_good / n_total;
    
    entropy_quality = entropy(p_total_good, p_total_mid, p_total_bad);
    info_gain = entropy_quality - (p_low * entropy_low + p_high * entropy_high);
    
    % Gini
    gini_low = gini(p_low_good, p_low_mid, p_low_bad);
    gini_high = gini(p_high_good, p_high_mid, p_high_bad);
    gini_split = p_low * gini_low + p_high * gini_high;
    
    fprintf('Entropy (low): %.3f, Entropy (high): %.3f\n', entropy_low, entropy_high);
    fprintf('Info Gain: %.3f\n', info_gain);
    fprintf('Gini (low): %.3f, Gini (high): %.3f\n', gini_low, gini_high);
    fprintf('Gini (split): %.3f\n', gini_split);
end