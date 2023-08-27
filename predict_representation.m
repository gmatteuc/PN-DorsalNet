function [optcoeffs,optcost] = predict_representation(target_mat, predictor_mat, regpar)

% get the number of neurons and stimulus conditions
[N, ~] = size(target_mat);
[M, ~] = size(predictor_mat);

% initialize the coefficients
optcoeffs = rand(M, N);

% define the options for fminunc
options = optimoptions('fminunc','Algorithm','quasi-newton','Display','off');
% options = optimoptions('fmincon','Algorithm','interior-point','Display','off');

% define the cost function
regularizerFun = @(X, lambda) nanmean( lambda*sum(abs(X), 1)' );
costFunUnreg = @(X) nanmean( nanmean((target_mat - linear_model(predictor_mat, X)).^2, 2) );
costFun = @(X) costFunUnreg(X) + regularizerFun(X, regpar);

% use fminunc to minimize the cost function
optcoeffs = fminunc(costFun, optcoeffs, options);
% [optcoeffs, ~] = fmincon(costFun, optcoeffs, [], [], [], [], [], [], [], options);

% output optimal cost
optcost = costFunUnreg(optcoeffs);

end