function rearranged_responses = recenter_neuron_responses(neuron_responses, target_center_idx)
% get the size of the input matrix
[N, ~] = size(neuron_responses);
% initialize rearranged_responses to be the same size as neuron_responses
rearranged_responses = zeros(size(neuron_responses));
% iterate over neurons
for i = 1:N
    % separate the first 12 stimuli and the next 12 stimuli
    first_half = neuron_responses(i, 1:12);
    second_half = neuron_responses(i, 13:24);
    % find the max of the first 12 stimuli and its index
    [~, max_idx] = max(first_half);
    % calculate the shift amount required to recenter the peak at target_center_idx
    shift_amount = target_center_idx - max_idx;
    % apply circular shift to first half
    first_half_shifted = circshift(first_half, [0, shift_amount]);
    % apply the same circular shift to the second half
    second_half_shifted = circshift(second_half, [0, shift_amount]);
    % combine the shifted halves back into one row
    rearranged_responses(i, :) = [first_half_shifted, second_half_shifted];
end
end

