function [avg_cost_crossval, std_cost_crossval, predicted_mats] =...
    get_crossvalidated_cost_predict_representation_parallel(...
    target_mat, predictor_mat, regpar, num_folds)

% get the number of stimuli
[~, K] = size(target_mat);

% randomly permute the stimuli indices
permuted_indices = randperm(K);

% initialize fold split variables
folds = cell(1, num_folds);
target_mat_train = cell(1, num_folds);
predictor_mat_train = cell(1, num_folds);
target_mat_test = cell(1, num_folds);
predictor_mat_test = cell(1, num_folds);

% loop over folds
for i = 1:num_folds
    
    % get  test indeces
    folds{i} = permuted_indices(round((i-1)*K/num_folds+1):round(i*K/num_folds));
    % define the test indices
    test_indices = folds{i};
    % define the training indices
    train_indices = setdiff(permuted_indices, test_indices);
    % get the training and test data
    target_mat_train{i} = target_mat(:, train_indices);
    predictor_mat_train{i} = predictor_mat(:, train_indices);
    target_mat_test{i} = target_mat(:, test_indices);
    predictor_mat_test{i} = predictor_mat(:, test_indices);
    
end

% preallocate for the output
predicted_mats = nan(num_folds, size(target_mat, 1), K);

% initialize temporary cell arrays
temp_costs = zeros(1, num_folds);
temp_predicted_mats = cell(1, num_folds);

% loop over the folds
parfor i = 1:num_folds
    
    % train the model
    fitted_coeffs = predict_representation(target_mat_train{i}, predictor_mat_train{i}, regpar);
    
    % compute the cost on the test data
    costFunUnreg = @(X) nanmean( nanmean((target_mat_test{i} - linear_model(predictor_mat_test{i}, X)).^2, 2) );
    temp_costs(i) = costFunUnreg(fitted_coeffs);
    
    % store test predictions
    temp_predicted_mats{i} = linear_model(predictor_mat_test{i}, fitted_coeffs);
    
    % output message
    disp(['fold #',num2str(i),' completed'])
    
end

% assign temporary cell arrays to output
for i = 1:num_folds
    predicted_mats(i,:,folds{i}) = temp_predicted_mats{i};
end
costs = temp_costs;

% compute the average cost
avg_cost_crossval = nanmean(costs);

% compute the average cost
std_cost_crossval = nanstd(costs);