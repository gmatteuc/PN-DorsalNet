function M_resampled = resample_neurons(M, K)

% get the dimensions of the input matrix
[numNeurons, numActivations] = size(M);

% initialize the resampled matrix
M_resampled = zeros(K, numActivations);

% generate the resampled matrix
for i = 1:K
    % randomly select a neuron
    neuron = M(randi(numNeurons), :);
    % split the neuron activation vector into two halves
    firstHalf = neuron(1:ceil(end/2));
    secondHalf = neuron(ceil(end/2)+1:end);
    % apply a random circular shift to each half
    firstHalf = circshift(firstHalf, randi(length(firstHalf)));
    secondHalf = circshift(secondHalf, randi(length(secondHalf)));
    % concatenate the shifted halves back together
    M_resampled(i, :) = [firstHalf, secondHalf];
end

end