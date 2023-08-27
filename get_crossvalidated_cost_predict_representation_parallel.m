function [costs, predicted_mats] =...
    get_crossvalidated_cost_predict_representation_parallel(...
    target_mat, predictor_mat, regpar, num_folds)

% get the number of stimuli
[N, K] = size(target_mat);

% define the maximum size of the target matrix in terms of N
max_size = 100;

% determine the number of subproblems
Q = ceil(N / max_size);

% initialize the output variables
costs = [];
predicted_mats = cell(Q, 1);

% randomly permute the stimuli indices
permuted_indices = randperm(K);

% loop over the subproblems
for q = 1:Q
    
    % determine the indices for the current subproblem
    indices = ((q-1)*max_size+1):min(q*max_size, N);
    
    % extract the subproblem data
    target_mat_sub = target_mat(indices, :);
    
    % run the function on the subproblem data
    [costs_sub, predicted_mats{q}] =...
        get_crossvalidated_cost_predict_representation_parallel_sub(...
        target_mat_sub, predictor_mat, permuted_indices, regpar, num_folds);
    
    % store the results
    costs = [costs, costs_sub]; %#ok<AGROW>
    
end

% concatenate the results
predicted_mats = cat(2, predicted_mats{:});

end