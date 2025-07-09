function out = processData(attribute, entropy_quality, good_quality, bad_quality) 
    %Processes the data, getting its values for analysis and model
    %construction by its split 
    
    entropy = @(good, bad) -(bad*log2(bad) + good*log2(good));
    gini = @(good, bad) 1 - bad^2 - good^2;

    n_total = height(attribute);
    
    % Split on attribute
    att_mean = mean(attribute);
    
    %attribute values above/under its mean
    low = attribute <= att_mean; 
    high = attribute > att_mean;
    n_low = sum(low);
    n_high = sum(high);
    
    % Quality splits within each attribute group
    low_bad = sum(low & bad_quality);
    low_good = sum(low & good_quality);
    high_bad = sum(high & bad_quality);
    high_good = sum(high & good_quality);
    
    p_low = n_low / n_total;
    p_high = n_high / n_total;
    
    p_low_bad = low_bad / n_low;
    p_low_good = low_good / n_low;
    p_high_bad = high_bad / n_high;
    p_high_good = high_good / n_high;
    
    fprintf('Mean: %.3f\n', att_mean);
    fprintf('Below Mean: %d (%.3f), Above Mean: %d (%.3f)\n', n_low, p_low, n_high, p_high);
    fprintf('P(low & bad): %.3f, P(low & good): %.3f\n', p_low_bad, p_low_good);
    fprintf('P(high & bad): %.3f, P(high & good): %.3f\n\n', p_high_bad, p_high_good);
    
    % Entropy
    entropy_low = entropy(p_low_good, p_low_bad);
    entropy_high = entropy(p_high_good, p_high_bad);
    
    % Info Gain
    info_gain = entropy_quality - (p_low * entropy_low + p_high * entropy_high);
    
    % Gini
    gini_low = gini(p_low_good, p_low_bad);
    gini_high = gini(p_high_good, p_high_bad);
    gini_split = p_low * gini_low + p_high * gini_high;
    
    fprintf('Entropy (low): %.3f, Entropy (high): %.3f\n', entropy_low, entropy_high);
    fprintf('Info Gain: %.3f\n', info_gain);
    fprintf('Gini (low): %.3f, Gini (high): %.3f\n', gini_low, gini_high);
    fprintf('Gini (split): %.3f\n', gini_split);

