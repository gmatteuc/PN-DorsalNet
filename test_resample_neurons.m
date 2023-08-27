% define a 2D matrix
M = [1 2 3 4 5 6; 6 5 4 3 2 1; 1 3 5 2 4 6];

% define the number of pseudo-neurons to generate
K = 5;

% generate the resampled matrix
M_resampled = resample_neurons(M, K);