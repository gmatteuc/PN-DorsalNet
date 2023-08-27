function [avg_cost_crossval, std_cost_crossval, predicted_mats] = get_crossvalidated_cost_predict_representation(target_mat, predictor_mat, regpar, num_folds)

% get the number of stimuli
[~, K] = size(target_mat);

% randomly permute the stimuli indices
permuted_indices = randperm(K);

% split the permuted indices into folds
folds = cell(1, num_folds);
for i = 1:num_folds
    folds{i} = permuted_indices(round((i-1)*K/num_folds+1):round(i*K/num_folds));
end

% initialize the cost array
costs = zeros(1, num_folds);

% initialize prediction array
predicted_mats = NaN(num_folds, size(target_mat,1), size(target_mat,2));

% loop over the folds
for i = 1:num_folds
    
    tic
    
    % define the test indices
    test_indices = folds{i};
    % define the training indices
    train_indices = setdiff(permuted_indices, test_indices);
    
    % get the training and test data
    target_mat_train = target_mat(:, train_indices);
    predictor_mat_train = predictor_mat(:, train_indices);
    target_mat_test = target_mat(:, test_indices);
    predictor_mat_test = predictor_mat(:, test_indices);
    
    % train the model
    fitted_coeffs = predict_representation(target_mat_train, predictor_mat_train, regpar);
    
    % compute the cost on the test data
    costFunUnreg = @(X) nanmean( nanmean((target_mat_test - linear_model(predictor_mat_test, X)).^2, 2) );
    costs(i) = costFunUnreg(fitted_coeffs);
    
    % store test predictions
    predicted_mats(i,:,folds{i}) = linear_model(predictor_mat_test, fitted_coeffs);
    
    toc
    
end

% compute the average cost
avg_cost_crossval = nanmean(costs);

% compute the average cost
std_cost_crossval = nanstd(costs);

end