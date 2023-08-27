function predicted_mat = linear_model(predictor_mat,weight_mat)

% get predicted mat as product of predictors and weights
predicted_mat=(predictor_mat'*weight_mat)';

end

